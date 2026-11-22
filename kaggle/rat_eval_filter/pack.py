"""Code-only dataset quyenuit24/lifemate-app-filter-code for kaggle/rat_eval_filter: the current lifemate/ package
(with app_filter.py), overlaid by the job on the repo of quyenuit24/lifemate-rat-eval-input. No competition data.

  python kaggle/rat_eval_filter/pack.py   # -> work/kaggle_dataset_app_filter_code/, then `kaggle datasets create -p .` there
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_app_filter_code"

shutil.rmtree(OUT / "lifemate", ignore_errors=True)
shutil.copytree(ROOT / "lifemate", OUT / "lifemate", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-app-filter-code", "id": "quyenuit24/lifemate-app-filter-code",
                                                        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
print(OUT, sorted(p.name for p in (OUT / "lifemate").iterdir()))
