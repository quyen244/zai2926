"""Kaggle job: public_test submission with the RAT agent (Qwen3-4B + traj_rat LoRA), one shard of four.

Four kernels run at once (2 per account, each "GPU T4 x2"), kernel k answers samples[k::4]; kaggle/rat_public_submit/merge.py
joins their submission_shard*.jsonl into one submission.jsonl (634 lines) and validates it locally.
Input: private dataset <account>/lifemate-rat-public-pack (kaggle/rat_public_submit/pack.py):
  repo/   lifemate/ code + public_test split without audio (jsonl, catalog, environment, zaic_tools)
  adapter/ RAT LoRA (work/kaggle/rat_train/out/traj_rat/adapter), checked by sha256
  asr_public_test.jsonl  cached fine-tuned PhoWhisper ASR (1933 clips, from quyenuit24/lifemate-public-submit v4)
  keep_apps.json         e5 top-7 app filter on that ASR (computed by the same v4 run; no e5/transformers in this job)
Agent: vLLM one server per GPU, WORKERS_PER_GPU samples in flight per GPU, spoken input (normalise=False, as RAT was
trained), ID guard on, step cap 6, greedy, trained chat template (history_think_template).
Time limits: a sample gets no new LLM step after SAMPLE_S seconds; no sample starts after STOP_S of job time and no step
starts after HARD_S. A cut sample keeps the calls it already made (its row is still a consistent calls + state hash);
a sample never started (or failed) gets an empty answer (no call, untouched state hash).
DRYRUN=1 (local): no pip, no GPU, a stub LLM; DRYRUN_INPUT = folder holding the pack, DRYRUN_WORK = output folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARD, NSHARDS = 2, 4
MODEL = "Qwen/Qwen3-4B"
LORA = "traj_rat"
ADAPTER_SHA256 = "90b9ec57d2601099c7965353159a0789c834433bb41f590dd442607b3bb3d93a"
SPLIT = "public_test"
STEP_CAP = 6
WORKERS_PER_GPU = 6   # rat_eval_filter: 6 per GPU, mean 137.6 s/sample, vLLM queue never empty; more only thrashes the KV cache
REQ_TIMEOUT = 240     # one LLM request (prefill + decode of one step); longest step seen < 200 s
SAMPLE_S = 300        # per-sample budget: p90 250 s, p99 ~500 s in rat_eval_filter
STOP_S = 45 * 60      # job time after which no sample starts
HARD_S = 47 * 60      # job time after which no LLM step starts
PORTS = [8000, 8001]
DRY = os.environ.get("DRYRUN")
INPUT = Path(os.environ.get("DRYRUN_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("DRYRUN_WORK", "/kaggle/working"))
REPO = WORK / "repo" if DRY else Path("/tmp/repo")
NO_CALL = {"role": "assistant", "content": "", "tool_calls": []}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def one_path(pattern: str, test=lambda p: True) -> Path:
    found = [p for p in INPUT.rglob(pattern) if test(p)]
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} under {INPUT}, found {[str(p) for p in found]}")
    return found[0]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def start_servers(adapter: Path) -> tuple[list[subprocess.Popen], str]:
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL))  # once, before two servers race for the same cache
    jinja = snap / "chat_template.jinja"
    native = jinja.read_text(encoding="utf-8") if jinja.is_file() else json.loads((snap / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(port),
               "--max-model-len", "16384", "--gpu-memory-utilization", "0.92", "--dtype", "half", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-lora", "--max-lora-rank", "16", "--max-loras", "1", "--lora-modules", f"{LORA}={adapter}"]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(WORK / f"vllm_gpu{gpu}.log", "w"), stderr=subprocess.STDOUT))
    t0 = time.time()
    for gpu, port in enumerate(PORTS):
        while True:
            if procs[gpu].poll() is not None:
                raise SystemExit(f"vLLM on GPU {gpu} died:\n" + (WORK / f"vllm_gpu{gpu}.log").read_text(errors="replace")[-3000:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except OSError:
                time.sleep(5)
        log(f"vLLM GPU {gpu} :{port} ready after {time.time() - t0:.0f}s")
    return procs, native


class Budget:
    """The LLM seen by run_sample for one sample: after the deadline it answers "no call", so the loop ends with the
    calls already made instead of the sample being lost."""

    def __init__(self, llm, deadline: float):
        self.llm, self.deadline, self.cut = llm, deadline, False

    def chat_tools(self, system, messages, tools, **kw):
        if time.time() > self.deadline:
            self.cut = True
            return dict(NO_CALL)
        return self.llm.chat_tools(system, messages, tools, max_retries=1, **kw)


class Stub:
    """DRYRUN only: one sys call, then stop."""

    def chat_tools(self, system, messages, tools, **_):
        if any(m.get("role") == "tool" for m in messages):
            return dict(NO_CALL)
        name = next((t["function"]["name"] for t in tools if t["function"]["name"].startswith("sys.")), tools[0]["function"]["name"])
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "s", "type": "function", "function": {"name": name, "arguments": "{}"}}]}


def main() -> None:
    t_job = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    pack = one_path("keep_apps.json").parent
    code = one_path("agent.py", lambda p: p.parent.name == "lifemate" and (p.parent.parent / "data").is_dir())
    shutil.rmtree(REPO, ignore_errors=True)
    shutil.copytree(code.parent.parent, REPO)
    adapter = one_path("adapter_config.json").parent
    if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise SystemExit(f"{adapter} is not the RAT adapter (sha256 mismatch)")
    asr = {r["audio"]: r["hyp"] for r in jsonl(pack / "asr_public_test.jsonl")}
    keep = {k: set(v) for k, v in json.loads((pack / "keep_apps.json").read_text(encoding="utf-8")).items()}
    log(f"shard {SHARD}/{NSHARDS} | pack {pack} | repo {code.parent.parent} -> {REPO} | adapter {adapter}")

    sys.path.insert(0, str(REPO))
    os.environ["LIFEMATE_DATA_ROOT"] = str(REPO / "data" / "train+public_test")
    from lifemate.data import load_samples, tool_env
    samples = load_samples(SPLIT)
    if len(samples) != 634:
        raise SystemExit(f"{len(samples)} public_test samples, expected 634")
    need = {s["audio"] for s in samples} | {t["audio"] for s in samples for t in s.get("history") or [] if t.get("role") == "user" and t.get("audio")}
    if need - asr.keys() or {s["id"] for s in samples} - keep.keys():
        raise SystemExit(f"ASR misses {len(need - asr.keys())} clips, keep_apps misses {len({s['id'] for s in samples} - keep.keys())} samples")
    mine = samples[SHARD::NSHARDS]
    if DRY and DRY.isdigit():
        mine = mine[: int(DRY)]
    log(f"{len(samples)} samples, this shard {len(mine)}; {len(asr)} ASR clips")

    procs, template = [], None
    if not DRY:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=False)
        log(f"pip done at {time.time() - t_job:.0f}s")
        procs, native = start_servers(adapter)
        from lifemate.sft import history_think_template
        template = history_think_template(native)
    from lifemate.agent import run_sample
    from lifemate.llm import LLM

    cache = WORK / "cache"
    cache.mkdir(exist_ok=True)
    files = [cache / f"rat.{SPLIT}.shard{SHARD}.gpu{k}.jsonl" for k in range(len(PORTS))]
    done = {r["id"]: r for f in files for r in jsonl(f)}
    todo = [s for s in mine if s["id"] not in done]
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "skipped": 0, "cut": 0}

    def one(llm, f: Path, s: dict) -> None:
        start = time.time()
        if start - t_job > STOP_S:
            with lock:
                stats["skipped"] += 1
            return
        guard = Budget(llm, min(start + SAMPLE_S, t_job + HARD_S))
        replies = []
        try:
            row = run_sample(s, SPLIT, asr[s["audio"]], guard, asr, STEP_CAP,
                             trace=lambda e, d: e == "llm_reply" and replies.append(d["reply"]),
                             keep_apps=keep[s["id"]], id_guard=True, normalise=False)
        except Exception as e:
            with lock:
                stats["failed"] += 1
            log(f"FAILED {s['id']}: {type(e).__name__}: {str(e)[:300]}")
            return
        row.update(sec=round(time.time() - start, 1), cut=guard.cut, replies=replies)
        with lock:
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[s["id"]] = row
            stats["ok"] += 1
            stats["cut"] += guard.cut
            if stats["ok"] % 10 == 0:
                log(f"{stats['ok']}/{len(todo)} done at {time.time() - t_job:.0f}s ({stats['failed']} failed, {stats['cut']} cut)")

    def shard(k: int) -> None:
        llm = Stub() if DRY else LLM(model=LORA, base_url=f"http://127.0.0.1:{PORTS[k]}/v1", timeout=REQ_TIMEOUT, chat_template=template)
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda s: one(llm, files[k], s), todo[k::len(PORTS)]))

    t_run = time.time()
    log(f"agent starts at {t_run - t_job:.0f}s: {len(todo)} samples")
    threads = [threading.Thread(target=shard, args=(k,)) for k in range(len(PORTS))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(f"agent done in {time.time() - t_run:.0f}s: {stats}")
    for p in procs:
        p.terminate()

    lines = []
    for s in mine:
        r = done.get(s["id"]) or {"tool_calls": [], "state_hash": tool_env(SPLIT, s["id"]).get_state_hash()}
        lines.append({"id": s["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]})
    (WORK / f"submission_shard{SHARD}.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    rows = [done[s["id"]] for s in mine if s["id"] in done]
    report = {"shard": SHARD, "nshards": NSHARDS, "samples": len(mine), "answered": len(rows), **stats,
              "empty_answers": sum(not r["tool_calls"] for r in rows),
              "mean_sec": round(sum(r["sec"] for r in rows) / max(len(rows), 1), 1),
              "guard_blocks": sum(r.get("guard_blocks", 0) for r in rows),
              "agent_s": round(time.time() - t_run), "job_s": round(time.time() - t_job)}
    (WORK / f"report_shard{SHARD}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log("report", report)
    shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
