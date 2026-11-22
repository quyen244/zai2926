"""Private dataset <account>/lifemate-rat-public-pack for kaggle/rat_public_submit (one copy per Kaggle account):
repo/ (lifemate/ code + public_test split without audio), adapter/ (RAT LoRA), asr_public_test.jsonl (cached ASR),
keep_apps.json (e5 top-7 filter on that ASR), both from the quyenuit24/lifemate-public-submit v4 run.

  python kaggle/rat_public_submit/pack.py   # -> work/kaggle_dataset_rat_public_pack/{quyenuit24,tquynlnguyn}/
then in each folder: kaggle datasets create -p . --dir-mode zip   (with that account's credential)
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_rat_public_pack"
PUB = ROOT / "data" / "train+public_test" / "public_test"
ADAPTER = ROOT / "work" / "kaggle" / "rat_train" / "out" / "traj_rat" / "adapter"
V4 = ROOT / "work" / "kaggle" / "public_submit_v4"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")

for owner in ("quyenuit24", "tquynlnguyn"):
    out = OUT / owner
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "lifemate", out / "repo" / "lifemate", ignore=IGNORE)
    for name in ("public_test.jsonl", "public_test_tools_catalog.json", "public_test_environment", "zaic_tools"):
        src, dst = PUB / name, out / "repo" / "data" / "train+public_test" / "public_test" / name
        shutil.copytree(src, dst, ignore=IGNORE) if src.is_dir() else shutil.copy2(src, dst.parent.mkdir(parents=True, exist_ok=True) or dst)
    shutil.copytree(ADAPTER, out / "adapter")
    for name in ("asr_public_test.jsonl", "keep_apps.json"):
        shutil.copy2(V4 / name, out / name)
    (out / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-rat-public-pack", "id": f"{owner}/lifemate-rat-public-pack",
                                                           "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(owner, f"{sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.0f} MB")
