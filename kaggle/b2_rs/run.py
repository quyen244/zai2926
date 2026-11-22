"""Kaggle evaluation of the B2 trajectory adapter on CV fold 0 (309 labelled train examples)."""
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
SYSTEMS = {"Qwen3-4B-trajB2": "traj_b2"}
TRAINED_TEMPLATE = {"Qwen3-4B-trajB2"}
FILTER = ""
FILTERED = set()
ADAPTERS = {"traj_b2": ("lifemate-b2-train", "traj_b2")}
RUN = ["Qwen3-4B-trajB2"]
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


class SamplingLLM:
    def __init__(self, llm, temperature: float):
        self.llm, self.temperature = llm, temperature

    def chat_tools(self, system, messages, tools, **kwargs):
        return self.llm.chat_tools(system, messages, tools, temperature=self.temperature, **kwargs)


def main() -> None:
    t_job = time.time()
    procs = []
    unpack()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=True)
    procs = start_servers()
    sys.path.insert(0, str(REPO))
    from lifemate.agent import run_sample
    from lifemate.data import fuzzy_params, load_samples
    from lifemate.gold import replay
    from lifemate.llm import LLM
    from lifemate.scoring import score_split
    from lifemate.sft import history_think_template

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    template = history_think_template(tok.chat_template)
    check_template(tok, template)
    prompts = [json.loads(l) for l in (REPO / "work/rs_b2/prompts.rs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    expected = 2234
    if len(prompts) != expected or len({(r["id"], r["source"], r["variant"]) for r in prompts}) != expected:
        raise SystemExit(f"wrong prompt bundle: expected {expected} unique rows, got {len(prompts)}")
    by_id = {s["id"]: s for s in load_samples("train")}
    gold = {}
    for sid in sorted({r["id"] for r in prompts}):
        sample = by_id[sid]
        env, results = replay("train", sid, sample["label"]["tool_calls"])
        if results != sample["label"]["tool_results"] or not all(env.get_state().get(a) == v for a, v in sample["label"]["final_state"].items()):
            raise SystemExit(f"gold replay mismatch: {sid}")
        gold[sid] = {"tool_calls": sample["label"]["tool_calls"], "state_hash": env.get_state_hash()}
    fz = fuzzy_params("train")
    candidates = 4
    jobs = [(i, c) for i in range(len(prompts)) for c in range(candidates)]
    out = WORK / "rs_cache.jsonl"
    selected_path = WORK / "rs_selected.jsonl"
    lock = threading.Lock()
    completed = set()
    if out.is_file():
        completed = {(r["prompt_index"], r["candidate"]) for r in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip())}
    jobs = [j for j in jobs if j not in completed]
    log(f"{len(prompts)} prompts (v1 1176 + v2 1058), 4 candidates each; {len(jobs)} candidate rollouts pending; full tools")
    stats = {"ok": 0, "failed": 0}

    def one(llms, i, c):
        prompt = prompts[i]
        sample = by_id[prompt["id"]]
        try:
            pred = run_sample(sample, "train", prompt["user_text"], llms, None, STEP_CAP)
            score, _ = score_split({prompt["id"]: {"tool_calls": pred["tool_calls"], "state_hash": pred["state_hash"]}},
                                   {prompt["id"]: gold[prompt["id"]]}, fz)
            rec = {"prompt_index": i, "candidate": c, "id": prompt["id"], "source": prompt["source"],
                   "variant": prompt["variant"], "score": round(score, 6), "tool_calls": pred["tool_calls"],
                   "state_hash": pred["state_hash"]}
            with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["ok"] += 1
                if stats["ok"] % 100 == 0:
                    log(f"{stats['ok']}/{len(jobs)} candidates complete")
        except Exception as e:
            with lock:
                stats["failed"] += 1
            log(f"FAILED prompt {i} candidate {c}: {type(e).__name__}: {str(e)[:200]}")

    def shard(k):
        base = LLM(model="traj_b2", base_url=f"http://127.0.0.1:{PORTS[k]}/v1", timeout=900, chat_template=template)
        llm = SamplingLLM(base, 0.8)
        mine = jobs[k::2]
        log(f"GPU {k}: {len(mine)} candidate rollouts")
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda j: one(llm, *j), mine))
        log(f"GPU {k} shard finished")

    threads = [threading.Thread(target=shard, args=(k,)) for k in range(len(PORTS))]
    for th in threads: th.start()
    for th in threads: th.join()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    best = {}
    for r in rows:
        if r["prompt_index"] not in best or r["score"] > best[r["prompt_index"]]["score"]:
            best[r["prompt_index"]] = r
    selected = [r for r in best.values() if r["score"] >= 0.8]
    selected_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    report = {"model": MODEL, "adapter": "traj_b2", "prompts": len(prompts), "candidates_per_prompt": candidates,
              "candidate_count": len(rows), "selected_threshold": 0.8, "selected_count": len(selected),
              "failed": stats["failed"], "score_mean": round(sum(r["score"] for r in rows) / max(1, len(rows)), 4),
              "best_score_mean": round(sum(r["score"] for r in best.values()) / max(1, len(best)), 4),
              "wall_s": round(time.time() - t_job)}
    (WORK / "rs_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(report)
    for p in procs: p.terminate()


if __name__ == "__main__":
    main()
