"""Private dataset quyenuit24/lifemate-public-infer for kaggle/public_infer: lifemate/ code + the public_test split
(jsonl, catalog, environments, zaic_tools, audio) in repo.tar.gz, and the REC adapter in adapter/.

  python kaggle/public_infer/pack.py   # -> work/kaggle_dataset_public_infer/, then `kaggle datasets create -p . --dir-mode zip` there
"""
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_public_infer"
PUB = ROOT / "data" / "train+public_test" / "public_test"
ADAPTER = ROOT / "work" / "kaggle" / "recovery_train" / "out" / "traj_rec" / "adapter"


def skip(m: tarfile.TarInfo):
    return None if "__pycache__" in m.name or m.name.endswith(".pyc") else m


OUT.mkdir(parents=True, exist_ok=True)
with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
    tar.add(ROOT / "lifemate", arcname="lifemate", filter=skip)
    for name in ("public_test.jsonl", "public_test_tools_catalog.json", "public_test_environment", "zaic_tools", "audio"):
        tar.add(PUB / name, arcname=f"data/train+public_test/public_test/{name}", filter=skip)
shutil.rmtree(OUT / "adapter", ignore_errors=True)
shutil.copytree(ADAPTER, OUT / "adapter")
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-public-infer", "id": "quyenuit24/lifemate-public-infer",
                                                        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
print(f"repo.tar.gz {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.0f} MB, adapter {sum(p.stat().st_size for p in (OUT / 'adapter').rglob('*')) / 1e6:.0f} MB")
