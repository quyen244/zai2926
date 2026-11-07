"""Build the private Kaggle dataset `lifemate-data` that GPU jobs read (code + train data + ASR cache + results cache).

  python scripts/kaggle_pack.py            # writes work/kaggle_dataset/ (repo.tar.gz + dataset-metadata.json)
  .venv/Scripts/kaggle.exe datasets create  -p work/kaggle_dataset            # first upload (private)
  .venv/Scripts/kaggle.exe datasets version -p work/kaggle_dataset -m "msg"   # later uploads

One archive, unpacked by the job into /kaggle/working/repo, so lifemate.data finds everything at the same relative
paths as on the workstation. Audio is left out (B1 uses the cached PhoWhisper hypotheses). The results cache is
work/kaggle/cache/*.jsonl: rows merged back from earlier job outputs (scripts/kaggle_merge_cache.py), so a later job
skips every (model, variant, code version, sample) that is already done.
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "work" / "kaggle_dataset"
DATASET_ID = "quyenuit24/lifemate-data"
TRAIN = Path("data/train+public_test/train")
INCLUDE = [Path("lifemate"), TRAIN / "train.jsonl", TRAIN / "train_tools_catalog.json", TRAIN / "train_environment",
           TRAIN / "zaic_tools", Path("work/asr/phowhisper/train.jsonl"), Path("work/kaggle/cache")]


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (PROJECT / "work/kaggle/cache").mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in INCLUDE if not (PROJECT / p).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for p in INCLUDE:
            tar.add(PROJECT / p, arcname=p.as_posix(), filter=keep)
    meta = {"title": "lifemate-data", "id": DATASET_ID, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with tarfile.open(OUT / "repo.tar.gz") as tar:
        names = tar.getnames()
    cached = sorted(n for n in names if n.startswith("work/kaggle/cache/") and n.endswith(".jsonl"))
    print(f"{OUT / 'repo.tar.gz'}: {len(names)} entries, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB, cache files: {cached}")


if __name__ == "__main__":
    sys.exit(main())
