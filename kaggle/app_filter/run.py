"""Kaggle job: Qwen3-4B-AWQ on train with the tool list FILTERED to the top-5 installed apps (+ contacts, + sys),
data-sharded over the two T4s of a "GPU T4 x2" session. Same samples, model, step cap and agent as kaggle/baseline/;
the only change is run_sample(keep_apps=...), read from work/retrieval/app_filter.top5.json (scripts/app_filter.py:
RRF of BGE-M3 and BM25, computed offline, so no embedder runs here). The cache tag gets ".top5" so filtered results
never mix with the unfiltered baseline cache.

Flow: unpack the private dataset `lifemate-data` (code, train data, cached ASR, results cache) -> install vLLM ->
start one vLLM OpenAI server per GPU (GPU 0 :8000, GPU 1 :8001, Triton attention, hermes tool parser) -> take the
first LIMIT train samples x VARIANTS, drop what the cache already has, split the rest in two shards (even / odd) ->
both shards run at once, each against its own GPU, LLM calls in parallel inside a shard -> every finished sample is
appended to the cache file -> score each variant on the LIMIT samples -> /kaggle/working/{cache/, scores.json,
submissions/}. Download the output and merge the cache back (scripts/kaggle_merge_cache.py) before the next job.

B0 = variant "gold" (reference transcript), B1 = variant "asr" (cached PhoWhisper hypothesis).
Cache file: cache/<model>.<variant>.sc<step_cap>.<code hash>.jsonl; the hash covers the agent/prompt/normaliser/LLM
client code, so a code change starts a new cache file instead of mixing results.
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

LIMIT = 100                      # first N train samples (raise for the full run: the cache keeps finished ones)
VARIANTS = ["gold", "asr"]
MODEL = "Qwen/Qwen3-4B-AWQ"
STEP_CAP = 6
FILTER = "work/retrieval/app_filter.top5.json"
WORKERS_PER_GPU = 6              # concurrent samples per GPU (vLLM batches them)
PORTS = [8000, 8001]
DRY = os.environ.get("DRYRUN")  # local code check: no install, no GPU, the LLM replays the gold calls
WORK = Path(os.environ.get("DRYRUN_WORK", ".")) if DRY else Path("/kaggle/working")
REPO = Path(os.environ["DRYRUN_REPO"]) if DRY else WORK / "repo"
if DRY:
    LIMIT = int(DRY)


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


def start_servers() -> list[subprocess.Popen]:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)  # once, before two servers race for the same cache
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}  # own torch.compile cache per server: a shared one races (EOFError on GPU 1, 2026-11-07)
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(port),
               "--max-model-len", "16384", "--gpu-memory-utilization", "0.88", "--dtype", "half", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
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

    samples = load_samples("train")[:LIMIT]
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (REPO / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    out_cache = WORK / "cache"
    out_cache.mkdir(exist_ok=True)
    keep = json.loads((REPO / FILTER).read_text(encoding="utf-8"))["keep"]
    tag = f"{MODEL.split('/')[-1]}.%s.sc{STEP_CAP}.top5.{code_hash()}"
    files, done = {}, {}
    for v in VARIANTS:
        files[v] = out_cache / f"{tag % v}.jsonl"
        src = REPO / "work/kaggle/cache" / files[v].name
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()] if src.is_file() else []
        done[v] = {r["id"]: r for r in rows}
        files[v].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    jobs = [(v, s) for v in VARIANTS for s in samples if s["id"] not in done[v]]
    log(f"{len(samples)} samples x {VARIANTS}: {sum(len(d) for d in done.values())} cached, {len(jobs)} to run, code {tag % '*'}")

    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0}

    def one(llm: LLM, v: str, s: dict) -> None:
        text = s["audio_text"] if v == "gold" else asr[s["audio"]]
        replies = []
        t = time.time()
        try:
            row = run_sample(s, "train", text, GoldReplay(s) if DRY else llm, asr if v == "asr" else None, STEP_CAP,
                             trace=lambda e, d: e == "llm_reply" and replies.append(d["reply"]), keep_apps=set(keep[v][s["id"]]))
        except Exception as e:  # not cached: a later job retries it
            with lock:
                stats["failed"] += 1
            log(f"FAILED {v} {s['id']}: {type(e).__name__}: {str(e)[:200]}")
            return
        row.update(sec=round(time.time() - t, 1), replies=replies)
        with lock:
            with open(files[v], "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[v][s["id"]] = row
            stats["ok"] += 1
            if stats["ok"] % 10 == 0:
                log(f"{stats['ok'] + stats['failed']}/{len(jobs)} done ({stats['failed']} failed)")

    def shard(k: int) -> None:
        llm = LLM(model=MODEL, base_url=f"http://127.0.0.1:{PORTS[k]}/v1", timeout=900)
        mine = jobs[k::2]
        log(f"shard {k} -> GPU {k}: {len(mine)} runs")
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda j: one(llm, *j), mine))
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
    report = {"model": MODEL, "filter": FILTER, "limit": LIMIT, "step_cap": STEP_CAP, "code": tag % "*", "run_s": round(time.time() - t_run),
              "job_s": None, "failed": stats["failed"], "variants": {}}
    (WORK / "submissions").mkdir(exist_ok=True)
    for v in VARIANTS:
        preds = {s["id"]: done[v][s["id"]] for s in samples if s["id"] in done[v]}
        final, per = score_split({k: {"tool_calls": r["tool_calls"], "state_hash": r["state_hash"]} for k, r in preds.items()},
                                 {k: gold[k] for k in preds}, fz)
        with open(WORK / "submissions" / f"train{LIMIT}.{v}.jsonl", "w", encoding="utf-8") as f:
            for k, r in preds.items():
                f.write(json.dumps({"id": k, "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]}, ensure_ascii=False) + "\n")
        report["variants"][v] = {"n": len(preds), "score": round(final, 4),
                                 "hash_ok": sum(p.hash_ok for p in per.values()),
                                 "mean_sec": round(sum(r["sec"] for r in preds.values()) / max(1, len(preds)), 1),
                                 "per_sample": {k: round(p.score, 3) for k, p in per.items()}}
        log(f"{v}: score {final:.4f} on {len(preds)} samples, hash ok {report['variants'][v]['hash_ok']}")
    report["job_s"] = round(time.time() - t_job)
    (WORK / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in procs:
        p.terminate()
    if not DRY:
        shutil.rmtree(REPO, ignore_errors=True)  # keep the output small: only cache/, submissions/, scores.json, logs


if __name__ == "__main__":
    main()
