"""Kaggle job: QLoRA SFT for the plan ablation, both experiments at once on a "GPU T4 x2" session.

Flow: unpack the private dataset lifemate-data (code + work/sft/{plan,traj}.{train,val}.jsonl built by
scripts/plan_data.py) -> pip install unsloth (torch cu128, xformers, transformers/trl pinned) -> download Qwen/Qwen3-4B once ->
GPU 0 trains A (plan + trajectory), GPU 1 trains B (trajectory only), same settings (lifemate/sft_train.py), at the
same time, independently -> /kaggle/working/out/{plan,traj}/{adapter/, log.jsonl, stats.json, train.log}.
SMOKE = True runs 2 optimizer steps x 4 rows on the 8 longest rows per experiment (worst-case memory, tokens/s per row).
Backend: unsloth (lifemate/sft_train.py --backend unsloth), installed as in unsloth's Kaggle Qwen3 (4B) notebook.
The eval job (kaggle/plan_sft_eval/) reads this kernel's output as an input.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SMOKE = False
# v2 (2026-11-09): only A is retrained, on plans that write "?" for values not visible yet (scripts/plan_data.py step_text);
# B's data is byte-identical to v1, so its v1 adapter (kernel lifemate-plan-sft-train, version 5) is reused by the eval.
# This folder now pushes kernel lifemate-plan-sft-train-v2, so the v1 output stays available as an eval input.
DATAS = ["plan"]
MODEL = "Qwen/Qwen3-4B"
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")  # outside /kaggle/working: never part of the kernel output, even when the job fails


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
if src:
    with tarfile.open(src) as tar:
        tar.extractall(REPO)
else:
    d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/agent.py"))
    shutil.copytree(d, REPO, dirs_exist_ok=True)
# v2 data or nothing: plans of v2 write "?" for unseen values (99 test rows have one). A stale dataset version silently
# retrained v1 plans otherwise (the 2026-11-08 12:40 upload had not become a version when the job was pushed).
if "plan" in DATAS and "=?" not in (REPO / "work/sft/plan.test.jsonl").read_text(encoding="utf-8"):
    sys.exit("work/sft/plan.* is v1 data (no '=?' in any plan): the dataset version is stale")
# Install exactly as unsloth's Kaggle notebook "Qwen3 (4B) Tool_Calling" (github.com/unslothai/notebooks, 2026-11-08):
# torch for CUDA 12.8 + xformers (the image's CUDA 13 torch had no bitsandbytes binary), unsloth, pinned transformers/trl.
PIP = [sys.executable, "-m", "pip", "install", "-q"]
for cmd in (PIP + ["torch", "torchvision", "torchaudio", "xformers", "--index-url", "https://download.pytorch.org/whl/cu128"],
            PIP + ["unsloth"],
            PIP + ["--no-deps", "--upgrade", "torchao>=0.16.0"],
            PIP + ["transformers==4.57.6"],
            PIP + ["--no-deps", "trl==0.22.2"],
            PIP + ["protobuf==3.20.3"]):
    subprocess.run(cmd, check=False)
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
# fail in seconds, not after the model download: CUDA torch, a 4-bit matmul through bitsandbytes, xformers, unsloth
check = subprocess.run([sys.executable, "-c", (
    "import torch, bitsandbytes as bnb, xformers, transformers, unsloth;"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'bnb', bnb.__version__, 'xformers', xformers.__version__,"
    " 'transformers', transformers.__version__, 'unsloth', unsloth.__version__);"
    "w = torch.randn(64, 64, dtype=torch.float16, device='cuda');"
    "q, st = bnb.functional.quantize_4bit(w, quant_type='nf4');"
    "print('nf4 ok', bnb.functional.dequantize_4bit(q, st).shape)")], check=False)
if check.returncode:
    sys.exit("environment check failed (see above)")
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(MODEL)
extra = ["--max-steps", "2", "--accum", "4", "--limit", "8"] if SMOKE else ["--max-hours", "10.8"]  # smoke: the 8 longest rows; full: stop cleanly before the 12 h kill
procs = {}
for gpu, data in enumerate(DATAS):
    out = WORK / "out" / data
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.sft_train", "--data", data, "--out", str(out), "--model", MODEL] + extra
    procs[data] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
    log(f"started {data} on GPU {gpu}")
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    tails = {d: (WORK / "out" / d / "train.log").read_text(errors="ignore").strip().splitlines()[-1:] for d in procs}
    log(f"{time.time() - t0:.0f}s", {d: (t[0][-160:] if t else "") for d, t in tails.items()})
codes = {d: p.returncode for d, p in procs.items()}
log("exit codes", codes)
for d in procs:
    shutil.rmtree(WORK / "out" / d / "ckpt", ignore_errors=True)
    print(f"--- {d} train.log tail ---\n" + "\n".join((WORK / "out" / d / "train.log").read_text(errors="ignore").splitlines()[-25:]))
shutil.rmtree(REPO, ignore_errors=True)
sys.exit(max(codes.values()))
