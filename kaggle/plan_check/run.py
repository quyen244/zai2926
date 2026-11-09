"""Kaggle job: Qwen3-8B-AWQ labels which gold call checks the condition (plan rule R3), data-sharded over two T4s.

Flow: unpack lifemate-data (work/sft/plan_check.requests.jsonl, written by scripts/plan_branch.py --requests: system +
user prompt + JSON schema per train sample) -> install vLLM -> one vLLM OpenAI server per GPU (Qwen/Qwen3-8B-AWQ,
Triton attention backend, fp16, max len 5120, seed 0) -> requests split by position (GPU 0 even, GPU 1 odd), 16 in
flight per GPU, temperature 0, thinking off, output forced to the schema -> /kaggle/working/labels.jsonl (one row per
sample, written as it finishes, so a timed-out job keeps what is done) + run.json (counts, timing).
Validation and merging are local: python scripts/plan_branch.py --merge work/kaggle/plan_check/labels.jsonl
"""
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

MODEL = "Qwen/Qwen3-8B-AWQ"
PORTS = [8000, 8001]
IN_FLIGHT = 16
DRY = os.environ.get("DRYRUN")  # local check: no server, every reply is {"conditional": false, "check_call": null}
WORK = Path(os.environ.get("DRYRUN_WORK", ".")) if DRY else Path("/kaggle/working")
REPO = Path(os.environ["DRYRUN_REPO"]) if DRY else Path("/tmp/repo")  # outside /kaggle/working: never part of the kernel output, even when the job fails


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unpack() -> None:
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:
        d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/agent.py"))
        shutil.copytree(d, REPO, dirs_exist_ok=True)


def start_servers() -> list:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}  # own torch.compile cache per server: a shared one races (EOFError on GPU 1, 2026-11-07)
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(port),
               "--max-model-len", "5120", "--gpu-memory-utilization", "0.88", "--dtype", "half", "--seed", "0"]
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


def ask(port: int, req: dict) -> dict:
    if DRY:
        return {"conditional": False, "check_call": None}
    body = {"model": MODEL, "messages": [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["user"]}],
            "temperature": 0, "max_tokens": 40, "seed": 0, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "label", "schema": req["schema"], "strict": True}}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        d = json.load(f)
    return json.loads(d["choices"][0]["message"]["content"])


def main() -> None:
    t_job = time.time()
    procs = []
    if not DRY:
        unpack()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=False)
        procs = start_servers()
    reqs = [json.loads(l) for l in (REPO / "work/sft/plan_check.requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if DRY:
        reqs = reqs[: int(DRY)]
    out = WORK / "labels.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.is_file() else set()
    todo = [r for r in reqs if r["id"] not in done]
    log(f"{len(reqs)} requests, {len(done)} already labelled, {len(todo)} to run")
    lock, stats = threading.Lock(), {"ok": 0, "failed": 0}

    def one(port: int, req: dict) -> None:
        t = time.time()
        try:
            reply = ask(port, req)
        except Exception as e:  # not written: a later run retries it
            with lock:
                stats["failed"] += 1
            log(f"FAILED {req['id']}: {type(e).__name__}: {str(e)[:200]}")
            return
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": req["id"], "reply": reply, "sec": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 100 == 0:
                log(f"{stats['ok']}/{len(todo)} done")

    def shard(k: int) -> None:
        with ThreadPoolExecutor(IN_FLIGHT) as pool:
            list(pool.map(lambda r: one(PORTS[k], r), todo[k::2]))

    t_run = time.time()
    threads = [threading.Thread(target=shard, args=(k,)) for k in range(len(PORTS))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    run = {"model": MODEL, "requests": len(reqs), "ok": stats["ok"], "failed": stats["failed"], "run_s": round(time.time() - t_run), "job_s": round(time.time() - t_job)}
    (WORK / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    log("done", run)
    for p in procs:
        p.terminate()
    if not DRY:
        shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
