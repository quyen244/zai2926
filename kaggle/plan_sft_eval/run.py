"""Kaggle job: evaluate the plan ablation on the held-out test split (cv5 fold 0, 309 samples), data-sharded over two T4s.

Flow: unpack lifemate-data (code + work/sft/*.test.jsonl) and the adapters from the training kernel
(quyenuit24/lifemate-plan-sft-train output) -> one vLLM server per GPU with Qwen/Qwen3-4B FP16 and both LoRA adapters
(--enable-lora: model "plan" = A plan + trajectory SFT, "traj" = B trajectory-only SFT, base name = the untuned
reference) -> runs (model, variant, sample), the rest split in two shards by position, both GPUs at once -> per-sample
result cache, same format as kaggle/baseline/ -> /kaggle/working/{cache/, scores.json}.
A and B are served with lifemate.sft.history_think_template (the template their training text was rendered with:
earlier steps of the turn keep the empty think block), checked on both servers before any run; the base keeps its own.
Variant "gold" = reference transcript on all 309 test samples (B0). Error recovery is measured offline on the natural
tool failures of these runs (scripts/ablation_metrics.py); no failure is injected.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
SYSTEMS = {"Qwen3-4B": MODEL, "Qwen3-4B-planSFT": "plan", "Qwen3-4B-trajSFT": "traj", "Qwen3-4B-planSFT2": "plan2",
           "Qwen3-4B-trajSFT-top5": "traj"}  # cache label -> vLLM model name
TRAINED_TEMPLATE = {"Qwen3-4B-planSFT", "Qwen3-4B-trajSFT", "Qwen3-4B-planSFT2", "Qwen3-4B-trajSFT-top5"}  # served with the template they were trained on
# tool list cut to the top-5 installed apps (+ contacts, + sys) by RRF of BGE-M3 and BM25 (scripts/app_filter.py, computed
# offline for every train sample, so no embedder runs here); reference-text query
FILTER = "work/retrieval/app_filter.top5.json"
FILTERED = {"Qwen3-4B-trajSFT-top5"}
# vLLM name -> (training kernel, out/<dir>): A v1 and B from the v1 kernel, A v2 ("?" for unseen plan values) from v2.
ADAPTERS = {"plan": ("lifemate-plan-sft-train", "plan"), "traj": ("lifemate-plan-sft-train", "traj"), "plan2": ("lifemate-plan-sft-train-v2", "plan")}
# Systems this push evaluates (their adapters must be in a kernel source; cached runs are skipped).
# 2026-11-09: A v2 (plans with "?" for unseen values), no filter, against the cached base / A v1 / B / B + top-5.
RUN = ["Qwen3-4B-planSFT2"]
VARIANTS = ["gold"]
STEP_CAP = 6
WORKERS_PER_GPU = 6              # concurrent samples per GPU (vLLM batches them)
PORTS = [8000, 8001]
DRY = os.environ.get("DRYRUN")  # local code check: no install, no GPU, the LLM replays the gold calls
WORK = Path(os.environ.get("DRYRUN_WORK", ".")) if DRY else Path("/kaggle/working")
REPO = Path(os.environ["DRYRUN_REPO"]) if DRY else WORK / "repo"
LIMIT = int(DRY) if DRY else 0
STOP_S = 10.5 * 3600  # start no new run after this much job time: Kaggle kills at 12 h and keeps nothing; done runs stay cached


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unpack() -> None:
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:  # Kaggle may have unpacked the archive itself
        d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/agent.py"))
        shutil.copytree(d, REPO, dirs_exist_ok=True)
    log("repo at", REPO, sorted(p.name for p in REPO.iterdir()))


def adapters() -> dict[str, Path]:
    return {k: next(p for p in Path("/kaggle/input").rglob(f"out/{d}/adapter/adapter_config.json") if p.parents[3].name == kernel).parent
            for k, (kernel, d) in ADAPTERS.items() if k in {SYSTEMS[x] for x in RUN}}


def start_servers() -> list[subprocess.Popen]:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)  # once, before two servers race for the same cache
    lora = [f"{k}={p}" for k, p in adapters().items()]
    log("adapters", lora)
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}  # own torch.compile cache per server: a shared one races (EOFError on GPU 1, 2026-11-07)
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(port),
               "--max-model-len", "12288", "--gpu-memory-utilization", "0.88", "--dtype", "half", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-lora", "--max-lora-rank", "16", "--max-loras", "3", "--lora-modules", *lora]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(WORK / f"vllm_gpu{gpu}.log", "w"), stderr=subprocess.STDOUT))
    t0 = time.time()
    for gpu, port in enumerate(PORTS):
        while True:
            if procs[gpu].poll() is not None:
                raise SystemExit(f"vLLM on GPU {gpu} died:\n" + (WORK / f"vllm_gpu{gpu}.log").read_text()[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except OSError:
                time.sleep(5)
        log(f"server GPU {gpu} :{port} ready after {time.time() - t0:.0f}s")
    return procs


def check_template(tok, template: str) -> None:
    """Fail before any run unless each server renders a request exactly as training did: for a test row with 3+ trained
    steps, the prompt token count vLLM reports (max_tokens 1) must equal the local count, with the model's own template
    and with the trained one, and the two must differ (proof that the per-request chat_template is applied)."""
    from lifemate.llm import tools_body
    from lifemate.sft import template_messages

    def probe(r: dict) -> bool:  # 3+ trained steps and Vietnamese text in an earlier call's arguments (JSON escaping shows)
        steps = [i for i, m in enumerate(r["messages"]) if m.get("train")]
        if len(steps) < 3:
            return False
        args = "".join(c["function"]["arguments"] for m in r["messages"][: steps[2]] for c in m.get("tool_calls") or [])
        return any(ord(ch) > 127 for ch in args)

    row = next(r for r in map(json.loads, (REPO / "work/sft/plan.test.jsonl").read_text(encoding="utf-8").splitlines()) if probe(r))
    j = [i for i, m in enumerate(row["messages"]) if m.get("train")][2]  # context of the third step: two earlier steps
    wire = [{k: v for k, v in m.items() if k != "train"} for m in row["messages"][:j]]
    local = template_messages(row)[: j + 1]
    for port in PORTS:
        counts = {}
        for name, t in (("native", None), ("trained", template)):
            want = len(tok(tok.apply_chat_template(local, tools=row["tools"], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False, **({"chat_template": t} if t else {})), add_special_tokens=False).input_ids)
            body = tools_body(next(SYSTEMS[x] for x in RUN if x in TRAINED_TEMPLATE), False, row["system"], wire, row["tools"], 1, 0, t)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            got = json.load(urllib.request.urlopen(req, timeout=300))["usage"]["prompt_tokens"]
            counts[name] = (got, want)
        log(f"template check :{port} {row['id']} step 3: {counts}")
        if any(g != w for g, w in counts.values()) or counts["native"][0] == counts["trained"][0]:
            raise SystemExit(f"vLLM on :{port} does not render the trained chat template: {counts}")


class GoldReplay:
    """DRYRUN only: answers with the sample's gold calls, one per turn (no model)."""

    def __init__(self, s: dict):
        self.calls = list(s["label"]["tool_calls"])

    def chat_tools(self, system, messages, tools, **_):
        if not self.calls:
            return {"role": "assistant", "content": "xong", "tool_calls": []}
        c = self.calls.pop(0)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "g", "type": "function", "function": {
            "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]}


def code_hash() -> str:
    h = hashlib.sha1()
    for f in ("agent.py", "prompt.py", "normalize.py", "llm.py", "data.py"):
        h.update((REPO / "lifemate" / f).read_bytes())
    return h.hexdigest()[:8]


def main() -> None:
    t_job = time.time()
    procs = []
    if not DRY:
        unpack()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=False)
        procs = start_servers()

    sys.path.insert(0, str(REPO))
    from lifemate.agent import run_sample
    from lifemate.data import fuzzy_params, load_samples
    from lifemate.gold import build_gold
    from lifemate.llm import LLM
    from lifemate.scoring import score_split
    from lifemate.sft import history_think_template

    template = None
    if not DRY:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        template = history_think_template(tok.chat_template)
        check_template(tok, template)

    test_ids = [json.loads(l)["id"] for l in (REPO / "work/sft/plan.test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {s["id"]: s for s in load_samples("train")}
    samples = [by_id[i] for i in sorted(test_ids)][: LIMIT or None]
    per_variant = {"gold": samples}
    out_cache = WORK / "cache"
    out_cache.mkdir(exist_ok=True)
    h = code_hash()
    files, done = {}, {}
    for label in RUN:
        for v in VARIANTS:
            files[(label, v)] = out_cache / f"{label}.{v}.sc{STEP_CAP}.{h}.jsonl"
            src = REPO / "work/kaggle/cache" / files[(label, v)].name
            rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()] if src.is_file() else []
            done[(label, v)] = {r["id"]: r for r in rows}
            files[(label, v)].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    jobs = [(label, v, s) for v in VARIANTS for s in per_variant[v] for label in RUN if s["id"] not in done[(label, v)]]
    log(f"{len(samples)} test samples; {sum(len(d) for d in done.values())} cached, {len(jobs)} to run, code {h}")

    keep = json.loads((REPO / FILTER).read_text(encoding="utf-8"))["keep"] if FILTERED & set(RUN) else {}
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "skipped": 0}

    def one(llms: dict, label: str, v: str, s: dict) -> None:
        if time.time() - t_job > STOP_S:
            with lock:
                stats["skipped"] += 1
            return
        replies = []
        t = time.time()
        try:
            row = run_sample(s, "train", s["audio_text"], GoldReplay(s) if DRY else llms[label], None, STEP_CAP,
                             trace=lambda e, d: e == "llm_reply" and replies.append(d["reply"]),
                             keep_apps=set(keep[v][s["id"]]) if label in FILTERED else None)
        except Exception as e:  # not cached: a later job retries it
            with lock:
                stats["failed"] += 1
            log(f"FAILED {label} {v} {s['id']}: {type(e).__name__}: {str(e)[:200]}")
            return
        row.update(sec=round(time.time() - t, 1), replies=replies)
        with lock:
            with open(files[(label, v)], "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[(label, v)][s["id"]] = row
            stats["ok"] += 1
            if stats["ok"] % 15 == 0:
                log(f"{stats['ok'] + stats['failed']}/{len(jobs)} done ({stats['failed']} failed)")

    def shard(k: int) -> None:
        llms = {label: LLM(model=name, base_url=f"http://127.0.0.1:{PORTS[k]}/v1", timeout=900,
                           chat_template=template if label in TRAINED_TEMPLATE else None) for label, name in SYSTEMS.items() if label in RUN}
        mine = jobs[k::2]
        log(f"shard {k} -> GPU {k}: {len(mine)} runs")
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda j: one(llms, *j), mine))
        log(f"shard {k} finished")

    t_run = time.time()
    threads = [threading.Thread(target=shard, args=(k,)) for k in range(len(PORTS))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(f"all shards done in {time.time() - t_run:.0f}s: {stats}")

    gold = build_gold("train")
    fz = fuzzy_params("train")
    report = {"model": MODEL, "systems": {x: SYSTEMS[x] for x in RUN}, "filter": FILTER if FILTERED & set(RUN) else None, "step_cap": STEP_CAP, "code": h, "run_s": round(time.time() - t_run), "failed": stats["failed"], "skipped_by_time_limit": stats["skipped"], "runs": {}}
    for (label, v), d in done.items():
        preds = {i: r for i, r in d.items()}
        if not preds:
            continue
        final, per = score_split({k: {"tool_calls": r["tool_calls"], "state_hash": r["state_hash"]} for k, r in preds.items()}, {k: gold[k] for k in preds}, fz)
        report["runs"][f"{label}.{v}"] = {"n": len(preds), "score": round(final, 4), "hash_ok": sum(p.hash_ok for p in per.values()),
                                          "mean_sec": round(sum(r["sec"] for r in preds.values()) / len(preds), 1)}
        log(f"{label} {v}: score {final:.4f} on {len(preds)} samples, hash ok {report['runs'][f'{label}.{v}']['hash_ok']}")
    report["job_s"] = round(time.time() - t_job)
    (WORK / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in procs:
        p.terminate()
    if not DRY:
        shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
