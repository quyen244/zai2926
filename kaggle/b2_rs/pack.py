"""Package the selected RS prompts and only their train environments for the Kaggle rollout job."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_b2_rs"
DATA = ROOT / "data" / "train+public_test" / "train"
DATASET_ID = "tquynlnguyn/lifemate-b2-rs-input"


def main() -> None:
    prompts = [json.loads(l) for l in (ROOT / "work/rs_b2/prompts.rs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(prompts) != 2234:
        raise SystemExit(f"expected 2234 RS prompts, got {len(prompts)}")
    ids = {r["id"] for r in prompts}
    rows = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    repo = OUT / "repo"
    if repo.resolve().parent != OUT.resolve():
        raise SystemExit("unexpected output directory")
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(ROOT / "lifemate", repo / "lifemate", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dest = repo / "data/train+public_test/train"
    dest.mkdir(parents=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    shutil.copy2(DATA / "train_tools_catalog.json", dest / "train_tools_catalog.json")
    shutil.copytree(DATA / "zaic_tools", dest / "zaic_tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    envdst = dest / "train_environment"
    envdst.mkdir()
    for sid in sorted(ids):
        shutil.copytree(DATA / "train_environment" / sid, envdst / sid, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (repo / "work/rs_b2").mkdir(parents=True)
    shutil.copy2(ROOT / "work/rs_b2/prompts.rs.jsonl", repo / "work/rs_b2/prompts.rs.jsonl")
    (repo / "work/sft").mkdir(parents=True)
    shutil.copy2(ROOT / "work/sft/plan.test.jsonl", repo / "work/sft/plan.test.jsonl")
    (OUT / "repo.tar.gz").unlink(missing_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for p in repo.iterdir():
            tar.add(p, arcname=p.relative_to(repo).as_posix(), filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-b2-rs-input", "id": DATASET_ID,
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"Prepared {len(prompts)} prompts, {len(ids)} train environments, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
