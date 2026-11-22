"""Kaggle job (account quyenuit24): fine-tune intfloat/multilingual-e5-small as the app-filter embedder on GPU.

Data: quyenuit24/lifemate-embed-train-input (kaggle/embed_train/pack.py): repo.tar.gz = lifemate/ +
scripts/train_app_embedder.py + splits/cv5.json + work/asr/phowhisper_ft/train.jsonl + data/train+public_test/train/
{train.jsonl, train_tools_catalog.json}. Kaggle may hand the tar over unpacked: both layouts are handled.
Base weights come from the Hugging Face hub (internet on), not from the dataset.

Training: listwise InfoNCE over each sample's installed apps, 3792 (query, gold app) rows = folds 1-4, gold + ASR text.
Eval: Top-K + threshold grid on the 550 fold-0 queries (275 gold + 275 ASR), smallest config with >= 99% recall.
Output: /kaggle/working/out/app_embedder_e5/ (fine-tuned weights + best_config.json + train.log)
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")  # outside /kaggle/working: never part of the kernel output
OUT = WORK / "out" / "app_embedder_e5"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unpack() -> None:
    src = next((p for p in Path("/kaggle/input").rglob("repo.tar.gz")), None)
    if src:
        with tarfile.open(src) as tar:
            tar.extractall(REPO)
    else:  # Kaggle unpacked the tar: the dataset holds lifemate/ and scripts/ directly
        d = next(p.parent.parent for p in Path("/kaggle/input").rglob("scripts/train_app_embedder.py"))
        shutil.copytree(d, REPO, dirs_exist_ok=True)
    for need in ("lifemate/data.py", "scripts/train_app_embedder.py", "splits/cv5.json",
                 "work/asr/phowhisper_ft/train.jsonl", "data/train+public_test/train/train_tools_catalog.json"):
        if not (REPO / need).exists():
            raise SystemExit(f"missing {need} in the dataset")
    log(f"repo at {REPO}: {sum(1 for _ in REPO.rglob('*'))} files")


def main() -> None:
    unpack()
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    # Kaggle's image already has torch + transformers; upgrading torch there breaks its CUDA build, so nothing is installed
    cmd = [sys.executable, "-X", "utf8", "scripts/train_app_embedder.py", "--epochs", "5", "--batch-size", "64",
           "--lr", "2e-5", "--out-dir", str(OUT)]
    log("start:", " ".join(cmd[3:]))
    with open(OUT / "train.log", "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    log("exit code", p.wait())
    shutil.rmtree(REPO, ignore_errors=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
