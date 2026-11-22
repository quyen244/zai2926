from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")


def main():
    t0 = time.time()
    print("KAGGLE_INPUT", [str(p) for p in Path("/kaggle/input").glob("*")], flush=True)
    manifests = list(Path("/kaggle/input").glob("**/manifest.jsonl"))
    if not manifests:
        raise SystemExit("No manifest.jsonl found under /kaggle/input; verify the private dataset is attached")
    DATA = manifests[0].parent
    # Kaggle may expose ZIP uploads either as files or as extracted directories.
    if not (DATA / "audio").is_dir():
        archive = DATA / "audio.zip"
        if not archive.is_file():
            archive = next(DATA.glob("**/audio.zip"), None)
        if archive is None:
            raise SystemExit(f"audio files missing from Kaggle input {DATA}")
        extracted = WORK / "data"
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        (extracted / "manifest.jsonl").write_bytes((DATA / "manifest.jsonl").read_bytes())
        data_path = extracted
    else:
        data_path = DATA
    print("GPUS", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "accelerate>=1.2", "librosa", "soundfile"], check=True)
    env = {**os.environ, "ASR_DATA": str(data_path), "ASR_OUT": str(WORK / "out"), "MAX_STEPS": "12"}
    train_script = DATA / "train.py"
    if not train_script.is_file():
        raise SystemExit(f"Trainer script missing from Kaggle dataset: {train_script}")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", str(train_script)]
    proc = subprocess.run(cmd, env=env)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    stats = json.loads((WORK / "out" / "stats.json").read_text(encoding="utf-8"))
    stats["job_total_seconds"] = time.time() - t0
    (WORK / "out" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("JOB_TOTAL", stats["job_total_seconds"], "SEC_PER_STEP", stats["seconds_per_step"], flush=True)


if __name__ == "__main__":
    main()
