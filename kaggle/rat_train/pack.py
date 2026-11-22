"""Build the private Kaggle dataset for the RAT SFT (account tquynlnguyn): code + training rows only.

  python kaggle/rat_train/pack.py      # -> work/kaggle_dataset_rat/ (repo.tar.gz + dataset-metadata.json)
  (then, with that account's token)  kaggle datasets create -p .   # run inside work/kaggle_dataset_rat

Rows: work/sft_rat/traj.{train,val}.jsonl (scripts/rat_data.py: gold + ASR rows with rationales, empty rows, recovery
rows). The REC adapter is not packed: the job reads it from kernel tquynlnguyn/lifemate-recovery-train. No raw
competition file goes in.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "work" / "kaggle_dataset_rat"
DATASET_ID = "tquynlnguyn/lifemate-rat-data"
RAT = PROJECT / "work/sft_rat"


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def lines(p: Path) -> list[str]:
    return [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    cv = json.loads((PROJECT / "splits/cv5.json").read_text(encoding="utf-8"))
    train, val = lines(RAT / "traj.train.jsonl"), lines(RAT / "traj.val.jsonl")
    ids = {json.loads(x).get("source_id", json.loads(x)["id"]) for x in train}
    if any(cv.get(i) == 0 for i in ids) or ids & {json.loads(x)["id"] for x in val}:
        raise SystemExit("training rows overlap CV fold 0 or the validation rows")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        tar.add(PROJECT / "lifemate", arcname="lifemate", filter=keep)
        for arc, rows in (("work/sft_rat/traj.train.jsonl", train), ("work/sft_rat/traj.val.jsonl", val)):
            data = ("\n".join(rows) + "\n").encode("utf-8")
            info = tarfile.TarInfo(arc)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    meta = {"title": "lifemate-rat-data", "id": DATASET_ID, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with tarfile.open(OUT / "repo.tar.gz") as tar:
        names = tar.getnames()
    print(f"{OUT / 'repo.tar.gz'}: {len(names)} entries, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB; "
          f"train {len(train)}, val {len(val)}")


if __name__ == "__main__":
    main()
