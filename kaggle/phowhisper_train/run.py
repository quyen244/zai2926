"""Kaggle job (account tquynlnguyn): PhoWhisper-large LoRA, one epoch over two jobs (kaggle/phowhisper_train/train.py).

Inputs: dataset tquynlnguyn/lifemate-phowhisper-data (pack.py: real.zip/real.jsonl, synth1.zip/synth1.jsonl, train.py,
cv5.json) + kernel output tquynlnguyn/lifemate-asr-synth-full (TTS batch 2: audio.zip, manifest.jsonl).
Part 1 (kernel lifemate-phowhisper-train-p1): no checkpoint in /kaggle/input -> train the first half, save out/checkpoint.
Part 2 (kernel lifemate-phowhisper-train-p2, kernel_sources p1): resume that checkpoint -> out/adapter, out/stats.json.
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/asr")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unzip(z: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dst)
    log("extracted", z, "->", dst, sum(1 for _ in dst.rglob("*.wav")), "wav")
    return dst


def main():
    inp = Path("/kaggle/input")
    data = next(p.parent for p in inp.rglob("real.jsonl"))
    syn2_src = next(p.parent for p in inp.rglob("manifest.jsonl") if "asr-synth" in str(p))
    ckpt = next((p.parent for p in inp.rglob("checkpoint-*/trainer_state.json")), None)
    part = 2 if ckpt else 1
    log("dataset", data, "| batch 2", syn2_src, "| part", part, "| checkpoint", ckpt)
    real = unzip(data / "real.zip", TMP / "real") if (data / "real.zip").is_file() else data
    syn1 = unzip(data / "synth1.zip", TMP / "syn1") if (data / "synth1.zip").is_file() else data
    syn2 = syn2_src
    if (syn2_src / "audio.zip").is_file():  # kernel output keeps the zip; extract it beside a copy of the manifest
        syn2 = unzip(syn2_src / "audio.zip", TMP / "syn2")
        (syn2 / "manifest.jsonl").write_bytes((syn2_src / "manifest.jsonl").read_bytes())
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "accelerate>=1.2",
                    "librosa", "soundfile"], check=True)
    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DATA": str(data), "REAL_DIR": str(real), "SYN1_DIR": str(syn1), "SYN2_DIR": str(syn2),
           "OUT": str(out), "PART": str(part), "MAX_HOURS": "10.3"}
    if ckpt:
        env["CKPT"] = str(ckpt)
    with open(out / "train.log", "w") as f:
        p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", str(data / "train.py")],
                             env=env, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        while p.poll() is None:
            time.sleep(120)
            tail = (out / "train.log").read_text(errors="ignore").replace("\r", "\n").strip().splitlines()[-1:]
            log(f"{time.time() - t0:.0f}s", tail[0][-200:] if tail else "")
    log("exit code", p.returncode)
    print("--- train.log tail ---\n" + "\n".join((out / "train.log").read_text(errors="ignore").replace("\r", "\n").splitlines()[-30:]))
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
