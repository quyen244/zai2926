"""Merge result-cache files downloaded from a Kaggle job into work/kaggle/cache/ (the cache the next job reads).

  .venv/Scripts/kaggle.exe kernels output quyenuit24/lifemate-baseline-qwen3-4b -p work/kaggle/baseline
  python scripts/kaggle_merge_cache.py work/kaggle/baseline/cache
  python scripts/kaggle_pack.py && .venv/Scripts/kaggle.exe datasets version -p work/kaggle_dataset -m "cache"

Rows are keyed by sample id inside each file (file name = model, variant, step cap, code hash); an id already in
the local cache is kept as is, so merging the same output twice changes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "work" / "kaggle" / "cache"


def read(p: Path) -> dict[str, dict]:
    if not p.is_file():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def main(src: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(src).glob("*.jsonl")):
        have, new = read(CACHE / f.name), read(f)
        added = {k: v for k, v in new.items() if k not in have}
        with open(CACHE / f.name, "a", encoding="utf-8", newline="\n") as out:
            for r in added.values():
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{f.name}: {len(have)} cached + {len(added)} new = {len(have) + len(added)}")


if __name__ == "__main__":
    main(sys.argv[1])
