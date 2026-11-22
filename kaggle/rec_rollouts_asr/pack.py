"""Package code + folds 1-4 train environments + both PhoWhisper ASR outputs (base, LoRA) for kaggle/rec_rollouts_asr (no app filter)."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_dataset_rec_rollouts_asr"
DATA = ROOT / "data" / "train+public_test" / "train"
DATASET_ID = "tquynlnguyn/lifemate-rec-rollouts-asr-input"


def main() -> None:
    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    ids = {i for i, f in cv.items() if f != 0}
    if len(ids) != 1236:
        raise SystemExit(f"expected 1236 fold 1-4 samples, got {len(ids)}")
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
    for d in ("phowhisper", "phowhisper_ft"):  # base and LoRA ASR text, every train clip (final + history turns)
        (repo / "work/asr" / d).mkdir(parents=True)
        shutil.copy2(ROOT / "work/asr" / d / "train.jsonl", repo / "work/asr" / d / "train.jsonl")
    (repo / "work/kaggle/cache").mkdir(parents=True)  # v2 finished samples: the job only runs the rest
    for f in (ROOT / "work/kaggle/rec_rollouts_asr_v2/cache").glob("*.jsonl"):
        shutil.copy2(f, repo / "work/kaggle/cache" / f.name)
    (repo / "work/sft").mkdir(parents=True)
    shutil.copy2(ROOT / "work/sft/plan.test.jsonl", repo / "work/sft/plan.test.jsonl")  # check_template probes a test row
    (OUT / "repo.tar.gz").unlink(missing_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        for p in repo.iterdir():
            tar.add(p, arcname=p.relative_to(repo).as_posix(), filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-rec-rollouts-asr-input", "id": DATASET_ID,
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"Prepared {len(ids)} fold 1-4 train environments, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
