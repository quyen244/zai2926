"""Build a private Kaggle dataset containing only CV test fold 0 for B2 evaluation."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_b2_eval"
DATA = ROOT / "data" / "train+public_test" / "train"
DATASET_ID = "tquynlnguyn/lifemate-b2-eval-data"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    cv = json.loads((ROOT / "splits" / "cv5.json").read_text(encoding="utf-8"))
    ids = {sid for sid, fold in cv.items() if fold == 0}
    samples = [r for r in jsonl(DATA / "train.jsonl") if r["id"] in ids]
    if len(samples) != 309 or {r["id"] for r in samples} != ids:
        raise SystemExit(f"expected all 309 fold-0 IDs; got {len(samples)} rows for {len(ids)} IDs")
    eval_rows = [r for r in jsonl(ROOT / "work" / "sft" / "plan.test.jsonl") if r["id"] in ids]
    if {r["id"] for r in eval_rows} != ids:
        raise SystemExit("plan.test.jsonl does not exactly cover CV fold 0")
    filt = json.loads((ROOT / "work" / "retrieval" / "app_filter.top5.json").read_text(encoding="utf-8"))
    filt["keep"] = {variant: {sid: apps for sid, apps in rows.items() if sid in ids}
                     for variant, rows in filt["keep"].items()}
    if any(len(rows) != 309 for rows in filt["keep"].values()):
        raise SystemExit("top-5 filter is incomplete for fold 0")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "lifemate", OUT / "repo" / "lifemate", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "repo" / "data" / "train+public_test" / "train").mkdir(parents=True)
    data_out = OUT / "repo" / "data" / "train+public_test" / "train"
    (data_out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples), encoding="utf-8")
    shutil.copy2(DATA / "train_tools_catalog.json", data_out / "train_tools_catalog.json")
    shutil.copytree(DATA / "zaic_tools", data_out / "zaic_tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env_out = data_out / "train_environment"
    env_out.mkdir()
    for sid in ids:
        shutil.copytree(DATA / "train_environment" / sid, env_out / sid, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "repo" / "work" / "sft").mkdir(parents=True)
    (OUT / "repo" / "work" / "sft" / "plan.test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows), encoding="utf-8")
    (OUT / "repo" / "work" / "retrieval").mkdir(parents=True)
    (OUT / "repo" / "work" / "retrieval" / "app_filter.top5.json").write_text(
        json.dumps(filt, ensure_ascii=False), encoding="utf-8")
    (OUT / "repo.tar.gz").unlink(missing_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for path in (OUT / "repo").iterdir():
            tar.add(path, arcname=path.relative_to(OUT / "repo").as_posix(),
                    filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-b2-eval-data", "id": DATASET_ID,
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"Prepared {OUT}: {len(samples)} fold-0 samples, {len(ids)} ToolEnv directories, "
          f"{(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
