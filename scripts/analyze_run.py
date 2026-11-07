"""Analyse a cached Kaggle run: overall score per variant, where the points are lost, and which samples got 1 or 0.

  python scripts/analyze_run.py                                  # every cache file in work/kaggle/cache
  python scripts/analyze_run.py --variant gold --show 10         # one variant, list 10 ids per bucket

Reads only local files (work/kaggle/cache/*.jsonl + train labels); no model runs. Scores with lifemate.scoring,
the same scorer the job used. Per predicted call it says why it did not match:
  exact               same tool, same arguments as a gold call
  extra args          has every gold argument but adds arguments the gold call does not have
  other args          right tool, an argument value differs or is missing
  wrong tool          a tool that is not in the gold calls
  extra ask           sys.ask_missing_info that the gold does not have
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, fuzzy_params, load_samples  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.scoring import score_split  # noqa: E402


def call_kind(p: dict, gold_calls: list[dict]) -> str:
    same = [g for g in gold_calls if g["name"] == p["name"]]
    if not same:
        return "extra ask" if p["name"] == "sys.ask_missing_info" else "wrong tool"
    if any(g["arguments"] == p["arguments"] for g in same):
        return "exact"
    if any(all(p["arguments"].get(a) == b for a, b in g["arguments"].items()) for g in same):
        return "extra args"
    return "other args"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "work" / "kaggle" / "cache")
    ap.add_argument("--variant", help="gold or asr; default: every file")
    ap.add_argument("--show", type=int, default=5, help="ids listed per bucket")
    a = ap.parse_args()

    labels = {s["id"]: s for s in load_samples("train")}
    gold, fz = build_gold("train"), fuzzy_params("train")
    files = sorted(a.cache_dir.glob(f"*.{a.variant}.*.jsonl" if a.variant else "*.jsonl"))
    if not files:
        raise SystemExit(f"no cache file in {a.cache_dir}")
    for f in files:
        rows = {r["id"]: r for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        final, per = score_split(rows, {k: gold[k] for k in rows}, fz)
        n = len(per)
        print("=" * 90, f"\n{f.name}\n" + "=" * 90)
        print(f"samples {n} | SCORE {final:.2f} / 100 | hash ok {sum(p.hash_ok for p in per.values())}/{n}"
              f" | score 1: {sum(p.score >= 0.999 for p in per.values())} | partial: {sum(0 < p.score < 0.999 for p in per.values())}"
              f" | score 0: {sum(p.score == 0 for p in per.values())} | mean sec/sample {sum(r.get('sec', 0) for r in rows.values()) / n:.0f}")
        kinds, n_pred, n_gold = Counter(), 0, 0
        for k, r in rows.items():
            gc = labels[k]["label"]["tool_calls"]
            n_gold += len(gc)
            n_pred += len(r["tool_calls"])
            kinds.update(call_kind(p, gc) for p in r["tool_calls"])
        print(f"calls: predicted {n_pred} vs gold {n_gold} ({n_gold - n_pred:+d} missing)")
        for kind, c in kinds.most_common():
            print(f"  {kind:12s} {c:4d}")
        buckets = {"score 1": [k for k, p in per.items() if p.score >= 0.999],
                   "partial": [k for k, p in per.items() if 0 < p.score < 0.999],
                   "score 0, hash ok": [k for k, p in per.items() if p.score == 0 and p.hash_ok],
                   "score 0, hash wrong": [k for k, p in per.items() if p.score == 0 and not p.hash_ok]}
        for name, ids in buckets.items():
            print(f"{name} ({len(ids)}): {' '.join(sorted(ids)[: a.show])}")


if __name__ == "__main__":
    main()
