"""Ids of the samples in given error groups of the cached run (definitions: lifemate/eda.py).

  python scripts/error_ids.py                                  # groups 1a 1b 2 3, both runs
  python scripts/error_ids.py --variant gold                   # only B0 (gold text)
  python scripts/error_ids.py --group extra_args wrong_value   # any group key from lifemate.eda.GROUPS
  python scripts/error_ids.py --only                           # samples whose arguments are the only problem in that group

Next step for one id: python scripts/compare_sample.py <id> --text gold|asr
Prints to the screen only (per-sample data stays off the repo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.eda import CACHE_DIR, GROUPS, TITLE, load_runs  # noqa: E402

DEFAULT = ["extra_args", "missing_args", "under", "over"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", default=DEFAULT, choices=[k for k, _ in GROUPS])
    ap.add_argument("--variant", choices=["gold", "asr", "all"], default="all")
    ap.add_argument("--only", action="store_true", help="1a/1b: keep samples whose closest calls have no other argument problem")
    ap.add_argument("--model", default="", help="lọc theo tên model trong tên file cache (vd Qwen3, QVikhr)")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    a = ap.parse_args()

    for (v, f), res in load_runs(a.cache_dir).items():
        tf = "gold" if ".gold." in f.name else "asr"
        if a.variant not in ("all", tf) or a.model.lower() not in f.name.lower():
            continue
        print(f"== {v}: {f.name} ({len(res)} mẫu)")
        for key in a.group:
            hit = [r for r in res if key in r["labels"]]
            if a.only and key in ("extra_args", "missing_args", "wrong_value"):
                other = {"extra_args", "missing_args", "wrong_value"} - {key}
                hit = [r for r in hit if not other & set(r["labels"])]
            print(f"-- {TITLE[key]}{' (chỉ lỗi này)' if a.only else ''}: {len(hit)} mẫu ({100 * len(hit) / len(res):.1f}%)")
            hit.sort(key=lambda r: r["id"])
            for i in range(0, len(hit), 5):
                print("   " + "   ".join(f"{r['id']} ({r['score']:.2f})" for r in hit[i:i + 5]))
        print(f"   so sánh 1 mẫu: python scripts/compare_sample.py <id> --text {tf} --model {f.name.split('.')[0]}\n")


if __name__ == "__main__":
    main()
