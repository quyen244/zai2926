"""How accurate are the model's tool calls in a cached run, at three strictness levels, overall and per tool.

  python scripts/tool_accuracy.py                              # every cache file in work/kaggle/cache
  python scripts/tool_accuracy.py --variant gold --top 25      # one variant, 25 most frequent gold tools

Levels (each predicted call is paired with at most one gold call of the same sample, maximum matching):
  name     right tool name, arguments ignored
  exact    what lifemate/scoring.py counts as a match: same argument set, values identical (fuzzy args token-F1 > 0.6)
  relaxed  every gold argument right, extra arguments allowed (the open question in TODO.md)
For each level: precision = matched / predicted calls, recall = matched / gold calls, F1.
Samples whose gold is empty are counted apart: correct when the model also made no call.
Reads local files only (cache + train labels); no model runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, fuzzy_params, load_samples  # noqa: E402
from lifemate.scoring import FUZZY_THRESHOLD, _args, calls_match, token_f1  # noqa: E402


def relaxed_match(p: dict, g: dict, fuzzy: dict) -> bool:
    if p.get("name") != g.get("name"):
        return False
    fz, pa = fuzzy.get(g["name"], set()), _args(p)
    for k, gv in _args(g).items():
        if k not in pa or (token_f1(pa[k], gv) <= FUZZY_THRESHOLD if k in fz else pa[k] != gv):
            return False
    return True


def matched_pairs(pred: list[dict], gold: list[dict], ok) -> list[tuple[int, int]]:
    """Maximum bipartite matching (Kuhn), returns (pred index, gold index) pairs."""
    adj = [[j for j, g in enumerate(gold) if ok(p, g)] for p in pred]
    owner = [-1] * len(gold)

    def aug(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if owner[j] == -1 or aug(owner[j], seen):
                    owner[j] = i
                    return True
        return False
    for i in range(len(pred)):
        aug(i, [False] * len(gold))
    return [(i, j) for j, i in enumerate(owner) if i != -1]


def prf(m: int, n_pred: int, n_gold: int) -> str:
    p, r = (m / n_pred if n_pred else 0.0), (m / n_gold if n_gold else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f"P {p:6.1%}  R {r:6.1%}  F1 {f:6.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "work" / "kaggle" / "cache")
    ap.add_argument("--variant", help="gold or asr; default: every file")
    ap.add_argument("--top", type=int, default=15, help="per-tool rows (most frequent gold tools)")
    a = ap.parse_args()

    labels = {s["id"]: s["label"]["tool_calls"] for s in load_samples("train")}
    fz = fuzzy_params("train")
    levels = {"name": lambda p, g: p.get("name") == g.get("name"), "exact": lambda p, g: calls_match(p, g, fz),
              "relaxed": lambda p, g: relaxed_match(p, g, fz)}
    files = sorted(a.cache_dir.glob(f"*.{a.variant}.*.jsonl" if a.variant else "*.jsonl"))
    if not files:
        raise SystemExit(f"no cache file in {a.cache_dir}")
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        tot = Counter()
        per_tool = defaultdict(Counter)  # tool -> gold, pred, and matched count per level
        empty_ok = empty_n = 0
        for r in rows:
            pred, gold = r["tool_calls"], labels[r["id"]]
            if not gold:
                empty_n += 1
                empty_ok += not pred
            tot["pred"] += len(pred)
            tot["gold"] += len(gold)
            for c in gold:
                per_tool[c["name"]]["gold"] += 1
            for c in pred:
                per_tool[c["name"]]["pred"] += 1
            for lv, ok in levels.items():
                pairs = matched_pairs(pred, gold, ok)
                tot[lv] += len(pairs)
                for _, j in pairs:
                    per_tool[gold[j]["name"]][lv] += 1
        print("=" * 100, f"\n{f.name}: {len(rows)} samples\n" + "=" * 100)
        print(f"calls: predicted {tot['pred']}, gold {tot['gold']}")
        for lv in levels:
            print(f"  {lv:8s} matched {tot[lv]:4d}   {prf(tot[lv], tot['pred'], tot['gold'])}")
        print(f"empty-gold samples: {empty_ok}/{empty_n} correctly made no call")
        print(f"\nper tool (top {a.top} by gold count; recall = matched / gold; 'pred' counts calls of that name)")
        print(f"  {'tool':34s} {'gold':>5s} {'pred':>5s} {'name R':>7s} {'exact R':>8s} {'relax R':>8s}")
        for name, c in sorted(per_tool.items(), key=lambda kv: (-kv[1]["gold"], kv[0]))[: a.top]:
            g = c["gold"] or 1
            print(f"  {name:34s} {c['gold']:5d} {c['pred']:5d} {c['name'] / g:7.0%} {c['exact'] / g:8.0%} {c['relaxed'] / g:8.0%}")
        never = sorted((n for n, c in per_tool.items() if not c["gold"]), key=lambda n: -per_tool[n]["pred"])
        if never:
            print("  called but never in gold: " + ", ".join(f"{n} ({per_tool[n]['pred']})" for n in never[:10]))


if __name__ == "__main__":
    main()
