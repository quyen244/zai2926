"""Kaggle job (account quyenuit24): public_test submission = fine-tuned PhoWhisper ASR + REC agent + e5 top-7 app filter.

Inputs:
  dataset quyenuit24/lifemate-public-infer (kaggle/public_infer/pack.py): repo.tar.gz = lifemate/ code +
      data/train+public_test/public_test/ (public_test.jsonl, catalog, environments, zaic_tools, audio); adapter/ = REC LoRA
      (work/kaggle/recovery_train/out/traj_rec/adapter, fold 0 gold 63.23)
  kernel quyenuit24/lifemate-asr-decode   -> out/phowhisper-ft-ct2 (PhoWhisper-large + LoRA, CTranslate2 float16)
  kernel quyenuit24/lifemate-embed-train  -> out/app_embedder_e5 (fine-tuned multilingual-e5-small)
Steps:
  1. ASR: every final and history user clip, 2 GPUs (one faster-whisper per GPU), decoding as kaggle/asr_decode
     (vi, beam 5, no previous-text conditioning; flagged hypothesis re-decoded once with peak norm + VAD).
  2. App filter on CPU: top-7 installed apps by the final ASR text (+ contacts; sys always offered), public_test catalog.
  3. Agent: Qwen3-4B + REC LoRA on vLLM (one server per GPU, samples sharded), trained chat template, normalised text
     (normalise=True, as REC was trained), ID guard on, step cap 6, greedy (vLLM seed 0, temperature 0 in lifemate.llm).
  4. submission.jsonl in public_test.jsonl order, validated with zaic_tools --validate.
Outputs (/kaggle/working): submission.jsonl, asr_public_test.jsonl, keep_apps.json, cache/*.jsonl (per-sample rows with
replies), report.json, vllm_gpu*.log.
"""
from __future__ import annotations

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
LORA = "traj_rec"
STEP_CAP = 6
WORKERS_PER_GPU = 6
PORTS = [8000, 8001]
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
TMP = Path("/tmp/asr")
SPLIT = "public_test"
STOP_S = 11 * 3600


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


# ------------------------------------------------------------------ ASR worker (one process per GPU)

def decode(fn, audio, **kw):
    segs, _ = fn(audio, language="vi", beam_size=5, condition_on_previous_text=False, **kw)
    segs = list(segs)
    return " ".join(s.text.strip() for s in segs).strip(), round(max((s.compression_ratio for s in segs), default=0.0), 3)


def asr_worker(gpu: int, tasks_file: str) -> None:
    import re
    import soundfile as sf
    from faster_whisper import WhisperModel
    hallu = re.compile(r"subscribe|đăng k[ýí]|ghiền mì gõ|cảm ơn các bạn đã (theo dõi|xem)|hẹn gặp lại|like|share|la la school", re.I)
    task = json.loads(Path(tasks_file).read_text(encoding="utf-8"))
    out = Path(task["out"])
    done = {r["audio"] for r in jsonl(out)}
    model = WhisperModel(task["model"], device="cuda", device_index=gpu, compute_type="float16")
    with open(out, "a", encoding="utf-8") as f:
        for k, rel in enumerate(c for c in task["clips"] if c not in done):
            x, sr = sf.read(str(Path(task["base"]) / rel), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                raise SystemExit(f"{rel}: sample rate {sr}")
            hyp, comp = decode(model.transcribe, x)
            redo = False
            if not hyp.strip() or hallu.search(hyp) or comp > 2.4:
                p = float(abs(x).max())
                h2, c2 = decode(model.transcribe, x * (10 ** (-3 / 20) / p) if p > 0 else x, vad_filter=True)
                if h2 and not (hallu.search(h2) or c2 > 2.4):
                    hyp, redo = h2, True
            f.write(json.dumps({"audio": rel, "hyp": hyp, "redo": redo}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 200 == 0:
                print(f"asr gpu{gpu}: {k}", flush=True)


def run_asr(samples: list[dict], base: Path) -> dict[str, str]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ctranslate2==4.8.4", "faster-whisper==1.2.1", "soundfile"], check=True)
    ct2 = next(p.parent for p in Path("/kaggle/input").rglob("model.bin") if p.parent.name == "phowhisper-ft-ct2")
    clips = [s["audio"] for s in samples] + [t["audio"] for s in samples for t in s.get("history") or []
                                             if t.get("role") == "user" and t.get("audio")]
    clips = list(dict.fromkeys(clips))
    miss = [c for c in clips if not (base / c).is_file()]
    if miss:
        raise SystemExit(f"{len(miss)} clips missing, e.g. {miss[:3]}")
    TMP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    import nvidia.cublas, nvidia.cudnn  # noqa: E401  (ctranslate2 needs libcublas / libcudnn 9 on the path)
    env["LD_LIBRARY_PATH"] = ":".join([str(Path(m.__path__[0]) / "lib") for m in (nvidia.cublas, nvidia.cudnn)] + [env.get("LD_LIBRARY_PATH", "")])
    procs = []
    for gpu in range(2):
        tf = TMP / f"tasks{gpu}.json"
        tf.write_text(json.dumps({"model": str(ct2), "base": str(base), "clips": clips[gpu::2], "out": str(TMP / f"asr{gpu}.jsonl")}), encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, __file__, "asr", str(gpu), str(tf)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"ASR workers exit codes {codes}")
    hyp = {r["audio"]: r["hyp"] for g in range(2) for r in jsonl(TMP / f"asr{g}.jsonl")}
    if set(clips) - hyp.keys():
        raise SystemExit(f"{len(set(clips) - hyp.keys())} clips without ASR text")
    (WORK / "asr_public_test.jsonl").write_text("".join(json.dumps({"audio": c, "hyp": hyp[c]}, ensure_ascii=False) + "\n" for c in clips), encoding="utf-8")
    log(f"ASR: {len(clips)} clips, {sum(not hyp[c].strip() for c in clips)} empty")
    return hyp


def app_filter_child(e5: Path) -> None:
    sys.path.insert(0, str(REPO))
    from lifemate.app_filter import AppFilter
    from lifemate.data import load_samples
    hyp = {r["audio"]: r["hyp"] for r in jsonl(WORK / "asr_public_test.jsonl")}
    filt = AppFilter(e5, split=SPLIT, device="cpu")
    keep = {s["id"]: sorted(filt.keep(hyp[s["audio"]], list(s["context"]["installed_apps"]))) for s in load_samples(SPLIT)}
    (WORK / "keep_apps.json").write_text(json.dumps(keep, ensure_ascii=False, indent=0), encoding="utf-8")


# ------------------------------------------------------------------ agent

def start_servers(adapter: Path) -> list[subprocess.Popen]:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    procs = []
    for gpu, port in enumerate(PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
               "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_gpu{gpu}"}
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(port),
               "--max-model-len", "16384", "--gpu-memory-utilization", "0.88", "--dtype", "half", "--seed", "0",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--enable-lora", "--max-lora-rank", "16", "--max-loras", "1", "--lora-modules", f"{LORA}={adapter}"]
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
        log(f"vLLM GPU {gpu} :{port} ready after {time.time() - t0:.0f}s")
    return procs


def main() -> None:
    t_job = time.time()
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:  # Kaggle unpacked the archive: copy the tree that holds lifemate/ and data/ (wav files included)
        d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/agent.py"))
        shutil.copytree(d, REPO, dirs_exist_ok=True)
    if not (REPO / "data/train+public_test/public_test/public_test.jsonl").is_file():
        raise SystemExit(f"public_test split missing under {REPO}: {sorted(p.name for p in REPO.iterdir())}")
    adapter = next(p.parent for p in Path("/kaggle/input").rglob("adapter_config.json")
                   if (p.parent / "adapter_model.safetensors").is_file() and "app_embedder" not in str(p))
    e5 = next(p.parent for p in Path("/kaggle/input").rglob("config.json") if p.parent.name == "app_embedder_e5")
    log("repo", REPO, "| adapter", adapter, "| e5", e5)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from lifemate.data import DATA_ROOT, load_samples
    samples = load_samples(SPLIT)
    order = [s["id"] for s in samples]
    log(f"{len(samples)} {SPLIT} samples")

    cached = next((p for p in Path("/kaggle/input").rglob("asr_public_test.jsonl")), None)
    if cached:  # an earlier version's ASR output (same model and decoding): no need to decode again
        asr = {r["audio"]: r["hyp"] for r in jsonl(cached)}
        need = {s["audio"] for s in samples} | {t["audio"] for s in samples for t in s.get("history") or [] if t.get("role") == "user" and t.get("audio")}
        if need - asr.keys():
            raise SystemExit(f"cached ASR {cached} misses {len(need - asr.keys())} clips")
        shutil.copy(cached, WORK / "asr_public_test.jsonl")
        log(f"ASR from cache {cached}: {len(asr)} clips")
    else:
        asr = run_asr(samples, DATA_ROOT / SPLIT)

    # in a child process: transformers must not be imported here before the vLLM pins replace it (v4 died on a mixed import)
    subprocess.run([sys.executable, __file__, "filter", str(e5)], cwd=REPO, check=True)
    keep = {k: set(v) for k, v in json.loads((WORK / "keep_apps.json").read_text(encoding="utf-8")).items()}
    log(f"app filter: {sum(map(len, keep.values()))} apps kept of {sum(len(s['context']['installed_apps']) for s in samples)} installed")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=False)
    procs = start_servers(adapter)
    from transformers import AutoTokenizer
    from lifemate.agent import run_sample
    from lifemate.llm import LLM
    from lifemate.sft import history_think_template
    template = history_think_template(AutoTokenizer.from_pretrained(MODEL).chat_template)

    cache = WORK / "cache"
    cache.mkdir(exist_ok=True)
    files = [cache / f"rec.{SPLIT}.asr_ft.e5top7.gpu{k}.jsonl" for k in range(len(PORTS))]
    done = {r["id"]: r for f in files for r in jsonl(f)}
    todo = [s for s in samples if s["id"] not in done]
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "skipped": 0}

    def one(llm, f: Path, s: dict) -> None:
        if time.time() - t_job > STOP_S:
            with lock:
                stats["skipped"] += 1
            return
        replies, t = [], time.time()
        try:
            row = run_sample(s, SPLIT, asr[s["audio"]], llm, asr, STEP_CAP,
                             trace=lambda e, d: e == "llm_reply" and replies.append(d["reply"]),
                             keep_apps=keep[s["id"]], id_guard=True, normalise=True)
        except Exception as e:
            with lock:
                stats["failed"] += 1
            log(f"FAILED {s['id']}: {type(e).__name__}: {str(e)[:300]}")
            return
        row.update(sec=round(time.time() - t, 1), replies=replies)
        with lock:
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[s["id"]] = row
            stats["ok"] += 1
            if stats["ok"] % 25 == 0:
                log(f"{stats['ok']}/{len(todo)} done ({stats['failed']} failed)")

    def shard(k: int) -> None:
        llm = LLM(model=LORA, base_url=f"http://127.0.0.1:{PORTS[k]}/v1", timeout=900, chat_template=template)
        with ThreadPoolExecutor(WORKERS_PER_GPU) as pool:
            list(pool.map(lambda s: one(llm, files[k], s), todo[k::2]))

    t_run = time.time()
    threads = [threading.Thread(target=shard, args=(k,)) for k in range(len(PORTS))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(f"agent done in {time.time() - t_run:.0f}s: {stats}")
    for p in procs:
        p.terminate()

    # samples that failed or were skipped: no call, untouched state (a valid line that scores what an empty answer scores)
    from lifemate.data import tool_env
    lines = []
    for sid in order:
        r = done.get(sid) or {"tool_calls": [], "state_hash": tool_env(SPLIT, sid).get_state_hash()}
        lines.append({"id": sid, "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]})
    sub = WORK / "submission.jsonl"
    sub.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    v = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", SPLIT, "--root", "..", "--validate", str(sub)],
                       cwd=DATA_ROOT / SPLIT, capture_output=True, text=True)
    log("validate rc", v.returncode, (v.stdout + v.stderr)[-1500:])
    rows = [done[i] for i in order if i in done]
    report = {"samples": len(order), "answered": len(rows), **stats, "empty_answers": sum(not r["tool_calls"] for r in rows),
              "mean_sec": round(sum(r["sec"] for r in rows) / max(len(rows), 1), 1),
              "guard_blocks": sum(r.get("guard_blocks", 0) for r in rows), "validate_rc": v.returncode,
              "job_s": round(time.time() - t_job)}
    (WORK / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log("report", report)
    shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "asr":
        asr_worker(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "filter":
        app_filter_child(Path(sys.argv[2]))
    else:
        main()
