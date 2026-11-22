"""Build the private Kaggle dataset for the recovery-row SFT (account tquynlnguyn): code + training rows only.

  python kaggle/recovery_train/pack.py      # -> work/kaggle_dataset_recovery/ (repo.tar.gz + dataset-metadata.json)
  (then, with that account's token)  kaggle datasets create -p .   # run inside work/kaggle_dataset_recovery

Rows: the 2115 B2 rows (work/sft_b2/traj.train.jsonl) + the 300 recovery rows (scripts/recovery_data.py), shuffled by
the trainer; val = B2's 60 rows. The B2 adapter is not packed: the job reads it from kernel tquynlnguyn/lifemate-b2-train,
the same output the B2 and ID-guard evals served. No raw competition file goes in.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "work" / "kaggle_dataset_recovery"
DATASET_ID = "tquynlnguyn/lifemate-recovery-data"
B2 = PROJECT / "work/sft_b2"
REC = PROJECT / "work/recovery/recovery.train.jsonl"


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def lines(p: Path) -> list[str]:
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    cv = json.loads((PROJECT / "splits/cv5.json").read_text(encoding="utf-8"))
    b2, rec, val = lines(B2 / "traj.train.jsonl"), lines(REC), lines(B2 / "traj.val.jsonl")
    if (len(b2), len(rec), len(val)) != (2115, 300, 60):
        raise SystemExit(f"unexpected counts b2={len(b2)} recovery={len(rec)} val={len(val)}")
    ids = {json.loads(l).get("source_id", json.loads(l)["id"]) for l in b2 + rec}
    if any(cv.get(i) == 0 for i in ids) or ids & {json.loads(l)["id"] for l in val}:
        raise SystemExit("training rows overlap CV fold 0 or the validation rows")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
        tar.add(PROJECT / "lifemate", arcname="lifemate", filter=keep)
        for arc, text in (("work/sft_rec/traj.train.jsonl", "\n".join(b2 + rec) + "\n"), ("work/sft_rec/traj.val.jsonl", "\n".join(val) + "\n")):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(arc)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    meta = {"title": "lifemate-recovery-data", "id": DATASET_ID, "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with tarfile.open(OUT / "repo.tar.gz") as tar:
        names = tar.getnames()
    print(f"{OUT / 'repo.tar.gz'}: {len(names)} entries, {(OUT / 'repo.tar.gz').stat().st_size / 1e6:.1f} MB; "
          f"train {len(b2)} + {len(rec)} = {len(b2) + len(rec)}, val {len(val)}")


if __name__ == "__main__":
    main()
