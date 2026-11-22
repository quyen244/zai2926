"""Kaggle job (account quyenuit24 copy of kaggle/phowhisper_train): PhoWhisper-large LoRA, one epoch over two jobs (kaggle/phowhisper_train/train.py).

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


def base_for(data: Path, manifest: str, tmp: Path, zipname: str) -> Path:
    """Directory under which the manifest's relative audio paths exist. Kaggle may keep a zip as a file (extract it to
    /tmp) or unpack it itself into a folder of the dataset (find which folder holds the first clip)."""
    if (data / zipname).is_file():
        return unzip(data / zipname, tmp)
    rel = json.loads((data / manifest).read_text(encoding="utf-8").splitlines()[0])["audio"]
    cands = [data] + [d for d in data.iterdir() if d.is_dir()] + [e for d in data.iterdir() if d.is_dir() for e in d.iterdir() if e.is_dir()]
    hit = next((d for d in cands if (d / rel).is_file()), None)
    if hit is None:
        raise SystemExit(f"{rel} not found under {data} (looked in {[str(c) for c in cands]})")
    log(manifest, "audio base", hit)
    return hit


def main():
    inp = Path("/kaggle/input")
    data = next(p.parent for p in inp.rglob("real.jsonl"))
    # quyenuit24 copy: batch 2 ships inside the dataset as synth2.zip + synth2.jsonl (no cross-account kernel output)
    syn2_src = data
    ckpt = next((p.parent for p in inp.rglob("checkpoint-*/trainer_state.json")), None)
    part = 2 if ckpt else 1
    log("dataset", data, "| batch 2", syn2_src, "| part", part, "| checkpoint", ckpt)
    real = base_for(data, "real.jsonl", TMP / "real", "real.zip")
    syn1 = base_for(data, "synth1.jsonl", TMP / "syn1", "synth1.zip")
    syn2 = base_for(data, "synth2.jsonl", TMP / "syn2", "synth2.zip")
    for base, man in ((real, "real.jsonl"), (syn1, "synth1.jsonl"), (syn2, "synth2.jsonl")):
        rels = [json.loads(l)["audio"] for l in (data / man).read_text(encoding="utf-8").splitlines() if l.strip()]
        miss = [r for r in rels if not (base / r).is_file()]
        log(f"{man}: {len(rels)} clips, missing {len(miss)}", miss[:3])
        if miss:
            raise SystemExit(f"{man}: {len(miss)} audio files missing under {base}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "accelerate>=1.2",
                    "librosa", "soundfile"], check=True)
    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DATA": str(data), "REAL_DIR": str(real), "SYN1_DIR": str(syn1), "SYN2_DIR": str(syn2), "SYN2_MANIFEST": str(data / "synth2.jsonl"),
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
