"""Compare the two SFT training rows of one sample: experiment B (trajectory only) | experiment A (plan + trajectory).

  python scripts/compare_sft.py train_0112                  # side by side, the turns the model is trained on
  python scripts/compare_sft.py 112 1346 --variant asr      # several ids (a bare number works), the ASR row (train split only)
  python scripts/compare_sft.py 112 --context               # also the prompt context: history turns, final user turn
  python scripts/compare_sft.py 112 --render                # the exact chat-template text each model trains on, the
                                                            # supervised (loss) spans marked >>> <<<, B then A
  python scripts/compare_sft.py --rule R3 --n 3             # random samples (seed 0) of one plan rule: R1 R2 R3 R4

Reads work/sft/{plan,traj}.{train,val,test}.jsonl (scripts/plan_data.py). The two rows share the system prompt, tool
schemas, history and gold calls; they differ only in the <plan> text of the trained assistant turns, which this
prints on its own line (">>> plan"). Token counts come from lifemate.sft.encode, i.e. what the trainer sees.
Screen only: per-sample data, never written to the repo.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT  # noqa: E402
from lifemate.sft import encode, render  # noqa: E402

SFT = PROJECT / "work" / "sft"
PLAN = re.compile(r"<plan>.*?</plan>", re.S)
W = 74  # column width


def load(d: str) -> dict[tuple[str, str], dict]:
    out = {}
    for part in ("train", "val", "test"):
        f = SFT / f"{d}.{part}.jsonl"
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[(r["id"], r["variant"])] = r
    return out


def args_text(a) -> str:
    return json.dumps(a if isinstance(a, dict) else json.loads(a or "{}"), ensure_ascii=False)


def turn_lines(m: dict) -> list[str]:
    """One trained message in the show_gt layout: plan, then the call or the final text; a tool result as '= {...}'."""
    if m["role"] == "tool":
        return [f"         = {m['content'][:300]}"]
    out = []
    content = m.get("content") or ""
    for p in PLAN.findall(content):
        out.append(f">>> plan {p}")
    rest = PLAN.sub("", content).strip()
    for c in m.get("tool_calls") or []:
        out.append(f"    -> {c['function']['name']} {args_text(c['function']['arguments'])}")
    if rest:
        out.append(f"    reply: {rest}")
    return out


def wrap(lines: list[str]) -> list[str]:
    out = []
    for l in lines:
        ind = len(l) - len(l.lstrip())
        out += textwrap.wrap(l, W, subsequent_indent=" " * (ind + 3), break_long_words=True, replace_whitespace=False) or [""]
    return out


def rule(row: dict) -> str:
    plans = [p for m in row["messages"] if m.get("train") for p in PLAN.findall(m.get("content") or "")]
    if any(p.endswith("-> replan</plan>") for p in plans):
        return "R3"
    if len(plans) > 1 and any("sys.ask_missing_info(" in p for p in plans):
        return "R2"
    return "R4" if not any(m.get("tool_calls") for m in row["messages"] if m.get("train")) else "R1"


def trained_tail(row: dict) -> list[dict]:
    """The final turn the model is trained on: from the first trained assistant message to the end (calls, results, reply)."""
    first = next(k for k, m in enumerate(row["messages"]) if m.get("train"))
    return row["messages"][first:]


def side_by_side(b: dict, a: dict) -> None:
    print(f"{'B: chỉ SFT (trajectory)':<{W}} | A: SFT + plan")
    print(f"{'-' * W}-+-{'-' * W}")
    for mb, ma in zip(trained_tail(b), trained_tail(a)):
        la_raw = turn_lines(ma)
        n_plan = sum(1 for l in la_raw if l.startswith(">>>"))
        lp, lrest = wrap(la_raw[:n_plan]), wrap(la_raw[n_plan:])
        lb = wrap(turn_lines(mb))
        for y in lp:  # plan lines: only in A
            print(f"{'':<{W}} |*{y}")
        for k in range(max(len(lb), len(lrest))):
            x = lb[k] if k < len(lb) else ""
            y = lrest[k] if k < len(lrest) else ""
            print(f"{x:<{W}} | {y}")
    print("(* = dòng chỉ có ở A; mọi dòng khác giống hệt nhau ở hai bản)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="train ids; a bare number works")
    ap.add_argument("--variant", default="gold", choices=["gold", "asr"], help="asr rows exist only for the train split")
    ap.add_argument("--context", action="store_true", help="also print the history and the final user turn")
    ap.add_argument("--render", action="store_true", help="print the exact chat-template text with the loss spans marked")
    ap.add_argument("--rule", choices=["R1", "R2", "R3", "R4"], help="pick random samples of this plan rule (with --n)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ", help="same Qwen3 tokenizer as training (local files only)")
    a = ap.parse_args()

    A, B = load("plan"), load("traj")
    ids = [x if x.startswith("train_") else f"train_{int(x):04d}" for x in a.ids]
    if a.rule:
        pool = sorted(i for (i, v), r in A.items() if v == a.variant and rule(r) == a.rule)
        ids += random.Random(0).sample(pool, min(a.n, len(pool)))
    if not ids:
        ap.error("give ids or --rule")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)

    for i in ids:
        ra, rb = A.get((i, a.variant)), B.get((i, a.variant))
        if ra is None or rb is None:
            print(f"\n{i}: no {a.variant} row (asr rows exist only for the train split)")
            continue
        ea, eb = encode(ra, tok, 10 ** 6), encode(rb, tok, 10 ** 6)
        user = next(m["content"] for m in reversed(ra["messages"]) if m["role"] == "user")
        print("\n" + "=" * (2 * W + 3))
        print(f"{i} | split {ra['split']} | {a.variant} | rule {rule(ra)} | {len(ra['tools'])} tools")
        print(f"  tokens       B {eb['n_tokens']:>6} | A {ea['n_tokens']:>6}  (+{ea['n_tokens'] - eb['n_tokens']})")
        print(f"  supervised   B {eb['n_supervised']:>6} | A {ea['n_supervised']:>6}  (+{ea['n_supervised'] - eb['n_supervised']})")
        print(f"  user: {user}")
        if a.context:
            print("  history (same in both):")
            for m in ra["messages"]:
                if m.get("train"):
                    break
                for l in wrap([f"    {m['role']}: {m.get('content') or ''}"] if not m.get("tool_calls") else turn_lines(m)):
                    print("  " + l)
        print("=" * (2 * W + 3))
        side_by_side(rb, ra)
        if a.render:
            for name, r in (("B: chỉ SFT", rb), ("A: SFT + plan", ra)):
                text, spans = render(r, tok)
                cut = text.rfind("<|im_start|>user", 0, spans[0][0])  # the final user turn (tool results are also "user" turns, later)
                body, off = text[cut:], cut
                for s, e in reversed(spans):
                    if s >= off:
                        body = body[: s - off] + ">>>" + body[s - off: e - off] + "<<<" + body[e - off:]
                print(f"\n----- {name}: chat template từ lượt user cuối (>>> <<< = phần tính loss) -----\n{body}")


if __name__ == "__main__":
    main()
