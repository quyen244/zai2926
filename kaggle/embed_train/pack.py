"""Build the private Kaggle dataset for embedding model training (account quyenuit24).

  python kaggle/embed_train/pack.py     # -> work/kaggle_dataset_embed_train/
  # then inside that dir:
  kaggle datasets create -p .           # first time
  kaggle datasets version -p . -m "v1" # subsequent

Contents:
  repo.tar.gz    - lifemate/ code + scripts/train_app_embedder.py + splits/cv5.json
                   + work/asr/phowhisper_ft/train.jsonl (asr_ft hypotheses)
                   + enriched catalog refs (data/train/catalog.json via lifemate.data)
  dataset-metadata.json
The base e5 weights are not packed: the job downloads them from the Hugging Face hub.

No raw audio or competition data included.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "work" / "kaggle_dataset_embed_train"
DATASET_ID = "quyenuit24/lifemate-embed-train-input"

ASR_FT = PROJECT / "work" / "asr" / "phowhisper_ft" / "train.jsonl"


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    skip = ("__pycache__", ".pyc", ".git", "node_modules")
    return None if any(s in info.name for s in skip) else info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- repo.tar.gz ---
    repo_path = OUT / "repo.tar.gz"
    with tarfile.open(repo_path, "w:gz") as tar:
        # lifemate package
        tar.add(PROJECT / "lifemate", arcname="lifemate", filter=keep)
        # train_app_embedder script
        script = PROJECT / "scripts" / "train_app_embedder.py"
        tar.add(script, arcname="scripts/train_app_embedder.py", filter=keep)
        # splits
        tar.add(PROJECT / "splits" / "cv5.json", arcname="splits/cv5.json")
        # asr_ft hypotheses (1.3 MB)
        tar.add(ASR_FT, arcname="work/asr/phowhisper_ft/train.jsonl")
        # train data (needed by lifemate.data.load_catalog + load_samples)
        data_train = PROJECT / "data" / "train+public_test" / "train"
        tar.add(data_train / "train_tools_catalog.json",
                arcname="data/train+public_test/train/train_tools_catalog.json")
        tar.add(data_train / "train.jsonl",
                arcname="data/train+public_test/train/train.jsonl")

    n = len(tarfile.open(repo_path).getnames())
    mb = repo_path.stat().st_size / 1e6
    print(f"repo.tar.gz: {n} entries, {mb:.1f} MB")

    # --- metadata ---
    meta = {"title": "lifemate-embed-train-input", "id": DATASET_ID,
            "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Dataset dir: {OUT}")
    total = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file()) / 1e6
    print(f"Total upload size: {total:.1f} MB")


if __name__ == "__main__":
    main()
