"""Pack RAT gold + paraphrase top-7 rows and trainer code for private Kaggle input."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/kaggle_dataset_rat_gold_para_top7"
DATA = ROOT / "work/sft_rat_toolonly_gold_para_top7"
DATASET_ID = "tquynlnguyn/lifemate-rat-gold-para-top7-data"


def main() -> None:
    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    train = [json.loads(x) for x in (DATA / "traj.train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    val = [json.loads(x) for x in (DATA / "traj.val.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(train) != 2690 or len(val) != 59:
        raise SystemExit("unexpected train/val row count")
    train_ids = {r["id"] for r in train}
    if train_ids & {r["id"] for r in val} or any(cv.get(i) == 0 for i in train_ids):
        raise SystemExit("train/val or fold-0 overlap")
    if any(r["variant"] not in {"gold", "para1", "para2"} for r in train) or any(r["variant"] != "gold" for r in val):
        raise SystemExit("unexpected variant")
    for row in train + val:
        offered = {t["function"]["name"] for t in row["tools"]}
        for msg in row["messages"]:
            if msg.get("train") is False or "<args>" in (msg.get("content") or ""):
                raise SystemExit("recovery or parameter reasoning row")
            if msg.get("train") and msg.get("tool_calls"):
                if "<select>" not in (msg.get("content") or ""):
                    raise SystemExit("tool-selection rationale missing")
                if any(call["function"]["name"] not in offered for call in msg["tool_calls"]):
                    raise SystemExit("gold tool hidden by top-7 filter")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        tar.add(ROOT / "lifemate", arcname="lifemate",
                filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
        for split in ("train", "val"):
            tar.add(DATA / f"traj.{split}.jsonl", arcname=f"work/sft_rat_toolonly_gold_para_top7/traj.{split}.jsonl")
        tar.add(DATA / "summary.json", arcname="work/sft_rat_toolonly_gold_para_top7/summary.json")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-rat-gold-para-top7-data",
        "id": DATASET_ID, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    with tarfile.open(OUT / "repo.tar.gz") as tar:
        names = set(tar.getnames())
    for split in ("train", "val"):
        if f"work/sft_rat_toolonly_gold_para_top7/traj.{split}.jsonl" not in names:
            raise SystemExit(f"missing {split} data from package")
    print(f"packed {len(train)} train, {len(val)} val, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
