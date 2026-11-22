"""Dataset quyenuit24/lifemate-dpo-data for kaggle/dpo_train: lifemate/ code + work/dpo/rec.{train,val}.jsonl (scripts/dpo_data.py,
pairs from train folds 1-4 only) in repo.tar.gz. Private dataset; nothing here is committed.

  python kaggle/dpo_train/pack.py   # -> work/kaggle_dataset_dpo/, then `kaggle datasets create|version -p .` there
"""
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_dpo"
cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
n = {}
for part in ("train", "val"):
    rows = [json.loads(l) for l in (ROOT / "work/dpo" / f"rat.{part}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(cv[r["id"]] == 0 for r in rows):
        raise SystemExit("a CV fold 0 sample in the DPO pairs")
    n[part] = len(rows)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "repo.tar.gz").unlink(missing_ok=True)
with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
    tar.add(ROOT / "lifemate", arcname="lifemate", filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
    for part in ("train", "val"):
        tar.add(ROOT / "work/dpo" / f"rat.{part}.jsonl", arcname=f"work/dpo/rat.{part}.jsonl")
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-dpo-data", "id": "quyenuit24/lifemate-dpo-data",
                                                        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
print(f"pairs {n}, repo.tar.gz {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB -> {OUT}")
