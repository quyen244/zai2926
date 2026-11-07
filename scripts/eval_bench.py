"""Score predictions on the hard-case benchmark (work/bench/hard_v1.jsonl) and say WHERE the model fails.

  python scripts/eval_bench.py --pred gold        # self-check: replays each gold through lifemate.agent.run_sample -> must be 100
  python scripts/eval_bench.py --pred empty       # reference floor: a model that never calls a tool
  python scripts/eval_bench.py --pred last-run    # the real train items the last Kaggle run already answered
  python scripts/eval_bench.py --pred work/bench/preds.qwen.jsonl   # any {"id", "tool_calls", "state_hash"} jsonl

Per item it records error labels that separate the failure stages (an item can carry several):
  decision     gold is [] but the model called, or gold needs calls and the model made none
  under        a gold tool name is missing from the prediction (fewer calls than needed)
  over         the prediction has a tool name the gold does not have (includes extra sys.ask_missing_info)
  lookup_skip  gold uses contacts.search / calendar.list_events / banking.list_beneficiaries and the prediction does not
  wrong_args   a tool is called by its right name but no call of it matches exactly (wrong, missing or extra arguments)
  hash         final state differs from gold (the score gate)
Score is lifemate.scoring.score_sample (same rule as the official metric as far as we know; see TODO.md on extra args).
Writes work/bench/eval.<pred>.json with the per-item labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, fuzzy_params  # noqa: E402
from lifemate.scoring import calls_match, score_sample  # noqa: E402

BENCH = PROJECT / "work" / "bench" / "hard_v1.jsonl"
LOOKUPS = ("contacts.search", "calendar.list_events", "banking.list_beneficiaries")
LABELS = ("decision", "under", "over", "lookup_skip", "wrong_args", "hash")


class GoldReplay:
    def __init__(self, calls):
        self.calls = list(calls)

    def chat_tools(self, system, messages, tools, **_):
        if not self.calls:
            return {"role": "assistant", "content": "xong", "tool_calls": []}
        c = self.calls.pop(0)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "g", "type": "function", "function": {
            "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]}


def labels_of(pred: list[dict], gold: list[dict], hash_ok: bool, fz) -> list[str]:
    out = []
    pn, gn = Counter(c["name"] for c in pred), Counter(c["name"] for c in gold)
    if bool(pred) != bool(gold):
        out.append("decision")
    if gn - pn:
        out.append("under")
    if pn - gn:
        out.append("over")
    if any(t in gn and t not in pn for t in LOOKUPS):
        out.append("lookup_skip")
    if any(not any(calls_match(p, g, fz) for p in pred if p["name"] == n for g in gold if g["name"] == n) for n in set(pn) & set(gn)):
        out.append("wrong_args")
    if not hash_ok:
        out.append("hash")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--pred", required=True, help="gold | empty | last-run | path to a predictions jsonl")
    a = ap.parse_args()

    items = [json.loads(l) for l in a.bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    fz = fuzzy_params("train")
    if a.pred == "gold":
        from lifemate.agent import run_sample
        preds = {}
        for it in items:
            s = {**it, "id": it["env_id"], "env_id": it["env_id"]}
            preds[it["id"]] = run_sample(s, "train", it["audio_text"], GoldReplay(it["label"]["tool_calls"]), None)
    elif a.pred == "empty":
        from lifemate.data import tool_env
        preds = {it["id"]: {"tool_calls": [], "state_hash": tool_env("train", it["env_id"]).get_state_hash()} for it in items}
    elif a.pred == "last-run":
        preds = {it["id"]: it["last_run"] for it in items if it.get("last_run")}
    else:
        preds = {r["id"]: r for r in (json.loads(l) for l in Path(a.pred).read_text(encoding="utf-8").splitlines() if l.strip())}
    run = [it for it in items if it["id"] in preds]
    if not run:
        raise SystemExit("no prediction matches a benchmark id")

    per_item, by = [], defaultdict(list)
    for it in run:
        p = preds[it["id"]]
        sc = score_sample(p["tool_calls"], p["state_hash"], it["label"]["tool_calls"], it["gold_hash"], fz)
        lab = labels_of(p["tool_calls"], it["label"]["tool_calls"], sc.hash_ok, fz)
        row = {"id": it["id"], "group": it["group"], "subtype": it["subtype"], "source": it["source"], "score": round(sc.score, 3), "labels": lab}
        per_item.append(row)
        by[("group", it["group"])].append(row)
        by[("subtype", it["subtype"])].append(row)
        by[("source", it["source"])].append(row)

    def line(name: str, rows: list[dict]) -> str:
        n = len(rows)
        c = Counter(l for r in rows for l in r["labels"])
        return (f"  {name:24s} {n:4d} {100 * sum(r['score'] for r in rows) / n:7.1f} {sum(r['score'] >= 0.999 for r in rows) / n:7.0%} "
                + " ".join(f"{c[l] / n:>11.0%}" for l in LABELS))
    head = f"  {'':24s} {'n':>4s} {'score':>7s} {'full':>7s} " + " ".join(f"{l:>11s}" for l in LABELS)
    print(f"pred = {a.pred}: {len(run)} of {len(items)} benchmark items")
    print(head)
    print(line("ALL", per_item))
    for kind in ("group", "source", "subtype"):
        print(f"-- by {kind}")
        for (k, v), rows in sorted(by.items()):
            if k == kind:
                print(line(v, rows))
    out = PROJECT / "work" / "bench" / f"eval.{Path(a.pred).stem if a.pred not in ('gold', 'empty', 'last-run') else a.pred}.json"
    out.write_text(json.dumps(per_item, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
