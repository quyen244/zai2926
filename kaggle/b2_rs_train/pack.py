"""Build a private Kaggle input for the two-GPU B2 RS mix ablation and fold-0 evaluation."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/kaggle_dataset_b2_rs_ablation"
DATASET_ID = "tquynlnguyn/lifemate-b2-rs-ablation-data"
DATA = ROOT / "data/train+public_test/train"
ADAPTER = ROOT / "work/kaggle/rs_train_source_adapter/out/traj_b2/adapter"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ignore(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def main() -> None:
    rs_only = ROOT / "work/rs_sft/rs_only.train.jsonl"
    rs_mix = ROOT / "work/rs_sft/rs_plus_v1.train.jsonl"
    val = ROOT / "work/rs_sft/rs_only.val.jsonl"
    eval_rows = ROOT / "work/sft/plan.test.jsonl"
    for path in (rs_only, rs_mix, val, eval_rows, ADAPTER / "adapter_model.safetensors", ROOT / "lifemate/sft_train.py"):
        if not path.exists():
            raise SystemExit(f"required input missing: {path}")
    if (len(read_jsonl(rs_only)), len(read_jsonl(rs_mix)), len(read_jsonl(val)), len(read_jsonl(eval_rows))) != (643, 2995, 60, 309):
        raise SystemExit("unexpected train/validation/test-fold counts")

    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    fold_ids = {sid for sid, fold in cv.items() if fold == 0}
    samples = [row for row in read_jsonl(DATA / "train.jsonl") if row["id"] in fold_ids]
    if len(samples) != 309 or {row["id"] for row in samples} != fold_ids:
        raise SystemExit("train data does not exactly cover 309 test-fold IDs")
    test_rows = read_jsonl(eval_rows)
    if {row["id"] for row in test_rows} != fold_ids:
        raise SystemExit("plan.test rows do not exactly cover test fold 0")
    train_ids = {row["source_id"] for row in read_jsonl(rs_only)}
    if train_ids & fold_ids:
        raise SystemExit("RS training prompts overlap test-fold IDs")

    repo = OUT / "repo"
    if repo.resolve().parent != OUT.resolve():
        raise SystemExit("unexpected package path")
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(ROOT / "lifemate", repo / "lifemate", ignore=ignore)
    shutil.copytree(ROOT / "work/rs_sft", repo / "work/rs_sft", ignore=ignore)
    shutil.copytree(ADAPTER, repo / "adapters/traj_b2", ignore=ignore)
    (repo / "work/sft").mkdir(parents=True)
    shutil.copy2(eval_rows, repo / "work/sft/plan.test.jsonl")
    data_out = repo / "data/train+public_test/train"
    data_out.mkdir(parents=True)
    (data_out / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    shutil.copy2(DATA / "train_tools_catalog.json", data_out / "train_tools_catalog.json")
    shutil.copytree(DATA / "zaic_tools", data_out / "zaic_tools", ignore=ignore)
    env_out = data_out / "train_environment"
    env_out.mkdir()
    for sid in sorted(fold_ids):
        shutil.copytree(DATA / "train_environment" / sid, env_out / sid, ignore=ignore)

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "repo.tar.gz"
    archive.unlink(missing_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for path in repo.iterdir():
            tar.add(path, arcname=path.relative_to(repo).as_posix(), filter=lambda item: None if "__pycache__" in item.name or item.name.endswith(".pyc") else item)
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
    required = {"lifemate/sft_train.py", "work/rs_sft/rs_only.train.jsonl", "work/rs_sft/rs_plus_v1.train.jsonl",
                "work/sft/plan.test.jsonl", "data/train+public_test/train/train.jsonl",
                "adapters/traj_b2/adapter_model.safetensors"}
    if not required.issubset(names):
        raise SystemExit(f"archive missing files: {sorted(required - names)}")
    metadata = {"title": "lifemate-b2-rs-ablation-data", "id": DATASET_ID, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"archive={archive} bytes={archive.stat().st_size} entries={len(names)}")
    print(f"RS-only=643; RS+all-v1=2995; val=60; test-fold=309; fold IDs overlap training=0")


if __name__ == "__main__":
    main()
