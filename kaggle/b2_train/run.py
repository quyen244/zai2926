"""Kaggle job (account tquynlnguyn): B2 = continue the SFT of B v1 on the paraphrased utterances, top-5 app tool lists.

Flow: unpack the private dataset tquynlnguyn/lifemate-b2-data (kaggle/b2_train/pack.py: lifemate/ code,
work/sft_b2/traj.{train,val}.jsonl from scripts/sft_filter_data.py, the B v1 adapter as adapters/traj_v1) -> check the
data (rewrites only, filtered tool lists) -> pip install unsloth exactly as kaggle/plan_sft_train/ -> download
Qwen/Qwen3-4B -> GPU 0: lifemate.sft_train --data traj --data-dir work/sft_b2 --init-adapter adapters/traj_v1, same
settings as B v1 otherwise (lr 1e-4 cosine, 3% warmup, accum 16, 1 pass, r16, max length 10240), val loss logged once
before the first step -> /kaggle/working/out/traj_b2/{adapter/, log.jsonl, stats.json, train.log}.
GPU 1 stays idle: one experiment. --max-hours 10.8 stops cleanly before the 12 h kill (expected ~5.2 h: 7.98M tokens at
the ~430 tokens/s B v1 trained at).
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")  # outside /kaggle/working: never part of the kernel output


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
if src:
    with tarfile.open(src) as tar:
        tar.extractall(REPO)
else:  # Kaggle unpacks an uploaded tar: the dataset holds lifemate/, work/, adapters/ directly
    d = next(p.parent.parent for p in Path("/kaggle/input").rglob("lifemate/sft_train.py"))
    shutil.copytree(d, REPO, dirs_exist_ok=True)
# the data must be the B2 rows: rewrites only (B v1's gold / asr rows left out) and top-5 tool lists
rows = [json.loads(l) for l in (REPO / "work/sft_b2/traj.train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bad = [r["variant"] for r in rows if not r["variant"].startswith("para")]
tools = sum(len(r["tools"]) for r in rows) / len(rows)
log(f"train rows {len(rows)}, variants not para: {len(bad)}, tools per row {tools:.1f}")
if bad or len(rows) != 2115 or tools > 25 or not (REPO / "adapters/traj_v1/adapter_model.safetensors").is_file():
    sys.exit("stale or wrong dataset version (expected 2115 rewrite rows, ~20 tools per row, adapters/traj_v1)")

# Install exactly as kaggle/plan_sft_train/run.py (unsloth's Kaggle notebook "Qwen3 (4B) Tool_Calling", 2026-11-08)
PIP = [sys.executable, "-m", "pip", "install", "-q"]
for cmd in (PIP + ["torch", "torchvision", "torchaudio", "xformers", "--index-url", "https://download.pytorch.org/whl/cu128"],
            PIP + ["unsloth"],
            PIP + ["--no-deps", "--upgrade", "torchao>=0.16.0"],
            PIP + ["transformers==4.57.6"],
            PIP + ["--no-deps", "trl==0.22.2"],
            PIP + ["protobuf==3.20.3"]):
    subprocess.run(cmd, check=False)
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
check = subprocess.run([sys.executable, "-c", (
    "import torch, bitsandbytes as bnb, xformers, transformers, unsloth, peft;"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'bnb', bnb.__version__, 'transformers', transformers.__version__,"
    " 'unsloth', unsloth.__version__, 'peft', peft.__version__);"
    "w = torch.randn(64, 64, dtype=torch.float16, device='cuda');"
    "q, st = bnb.functional.quantize_4bit(w, quant_type='nf4');"
    "print('nf4 ok', bnb.functional.dequantize_4bit(q, st).shape)")], check=False)
if check.returncode:
    sys.exit("environment check failed (see above)")
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(MODEL)
out = WORK / "out" / "traj_b2"
out.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO)}
cmd = [sys.executable, "-X", "utf8", "-m", "lifemate.sft_train", "--data", "traj", "--data-dir", "work/sft_b2",
       "--init-adapter", str(REPO / "adapters/traj_v1"), "--out", str(out), "--model", MODEL, "--max-hours", "10.8"]
p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
log("started traj_b2 on GPU 0")
t0 = time.time()
while p.poll() is None:
    time.sleep(60)
    tail = (out / "train.log").read_text(errors="ignore").strip().splitlines()[-1:]
    log(f"{time.time() - t0:.0f}s", tail[0][-160:] if tail else "")
log("exit code", p.returncode)
shutil.rmtree(out / "ckpt", ignore_errors=True)
print("--- train.log tail ---\n" + "\n".join((out / "train.log").read_text(errors="ignore").splitlines()[-25:]))
shutil.rmtree(REPO, ignore_errors=True)
sys.exit(p.returncode)
