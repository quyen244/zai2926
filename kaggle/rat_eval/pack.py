"""Package code + fold-0 train environments + PhoWhisper ASR outputs + RAT adapter for kaggle/rat_eval."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_rat_eval"
DATA = ROOT / "data" / "train+public_test" / "train"
ADAPTER_SRC = ROOT / "work" / "kaggle" / "rat_train" / "out" / "traj_rat" / "adapter"
DATASET_ID = "quyenuit24/lifemate-rat-eval-input"


def main() -> None:
    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    ids = {i for i, f in cv.items() if f == 0}
    if len(ids) != 309:
        raise SystemExit(f"expected 309 fold 0 samples, got {len(ids)}")
    if not (ADAPTER_SRC / "adapter_model.safetensors").is_file():
        raise SystemExit(f"missing RAT adapter in {ADAPTER_SRC}")

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
    (repo / "splits").mkdir(parents=True)
    shutil.copy2(ROOT / "splits/cv5.json", repo / "splits/cv5.json")
    for d in ("phowhisper", "phowhisper_ft"):
        (repo / "work/asr" / d).mkdir(parents=True)
        shutil.copy2(ROOT / "work/asr" / d / "train.jsonl", repo / "work/asr" / d / "train.jsonl")
    (repo / "work/sft").mkdir(parents=True)
    shutil.copy2(ROOT / "work/sft/plan.test.jsonl", repo / "work/sft/plan.test.jsonl")

    (OUT / "repo.tar.gz").unlink(missing_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for p in repo.iterdir():
            tar.add(p, arcname=p.relative_to(repo).as_posix(), filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)

    # Copy the RAT adapter directly into the dataset
    adapter_dst = OUT / "adapter"
    shutil.rmtree(adapter_dst, ignore_errors=True)
    shutil.copytree(ADAPTER_SRC, adapter_dst)

    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "lifemate-rat-eval-input",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}]
    }, indent=2), encoding="utf-8")

    repo_mb = (OUT / "repo.tar.gz").stat().st_size / 1e6
    adapter_mb = sum(f.stat().st_size for f in adapter_dst.rglob("*") if f.is_file()) / 1e6
    print(f"Prepared {len(ids)} fold-0 train environments (repo.tar.gz: {repo_mb:.1f} MB, adapter: {adapter_mb:.1f} MB)")


if __name__ == "__main__":
    main()
