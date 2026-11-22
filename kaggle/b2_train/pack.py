"""Build the private Kaggle dataset the B2 job reads (account tquynlnguyn): code + B2 data + the B v1 adapter only.

  python kaggle/b2_train/pack.py          # -> work/kaggle_dataset_b2/ (repo.tar.gz + dataset-metadata.json)
  (then, with that account's token)  kaggle datasets create -p .   # run inside work/kaggle_dataset_b2

Contents: lifemate/ (training code), work/sft_b2/traj.{train,val}.jsonl (scripts/sft_filter_data.py) and the B v1
adapter (kernel quyenuit24/lifemate-plan-sft-train, out/traj/adapter) as adapters/traj_v1/. No raw competition file
(train.jsonl, environments, catalog, audio) goes in: training needs only the rendered rows.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "work" / "kaggle_dataset_b2"
DATASET_ID = "tquynlnguyn/lifemate-b2-data"
ADAPTER = PROJECT / "work/kaggle/plan_sft_train/out/traj/adapter"
FILES = {"lifemate": PROJECT / "lifemate", "work/sft_b2/traj.train.jsonl": PROJECT / "work/sft_b2/traj.train.jsonl",
         "work/sft_b2/traj.val.jsonl": PROJECT / "work/sft_b2/traj.val.jsonl", "adapters/traj_v1": ADAPTER}


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def main() -> None:
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for arc, p in FILES.items():
            tar.add(p, arcname=arc, filter=keep)
    meta = {"title": "lifemate-b2-data", "id": DATASET_ID, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with tarfile.open(OUT / "repo.tar.gz") as tar:
        names = tar.getnames()
    print(f"{OUT / 'repo.tar.gz'}: {len(names)} entries, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB; "
          f"top level {sorted({n.split('/')[0] for n in names})}")


if __name__ == "__main__":
    main()
