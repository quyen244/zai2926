"""SFT rows for RAT (plan step 5): REC continued on trajectories whose every call carries an AutoTool-style rationale.

Each trained assistant step is
    <select>why this tool now <tool>app.func</tool> <args>how the vague values were converted</args></select>
followed by the native tool call (scripts/rationale_data.py wrote and checked the rationales, gpt-oss-120b). The agent
sees the user's spoken words (build_prompt(..., normalise=False): no number/time normaliser, the date hint line kept),
so the rationale is where it learns to convert "8 giờ kém 15 tối" to 19:45 and "hai triệu tám" to 2800000.

Rows (train folds 1-4 only, never CV fold 0; val = plan_data's 60 fold-1 samples, gold text):
  gold      one row per sample whose gold rationale passed every check
  asr       the same with the fine-tuned PhoWhisper text (final + history turns) and its own checked rationale
  empty     samples without any call: the final reply only (no rationale), gold and ASR text
  recovery  300 rows: before a lookup that supplies an ID, the write call with an invented ID (untrained) and the ID
            guard's error are inserted (scripts/recovery_data.py); the lookup's rationale then starts with why the call
            was blocked. Sources spread round-robin over apps; gold or ASR text at random.
Every row is replayed on a fresh ToolEnv (gold state hash) and rendered with the Qwen3 template under the length cap.

  python scripts/rat_data.py            # -> work/sft_rat/traj.{train,val}.jsonl, prints counts and token stats
  python scripts/rat_data.py --show 2   # also prints rows in the show_gt layout
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rationale_data as R  # noqa: E402
from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from lifemate.sft import encode  # noqa: E402
from plan_data import splits, tool_kinds, trained_turns  # noqa: E402
from recovery_data import body_start, check as recovery_check, make as make_recovery  # noqa: E402

OUT = PROJECT / "work" / "sft_rat"
TOOL_ONLY = False
MAX_LEN = 10240
ARROW = re.compile(r"\s*(?:→|⇒|➔|->|=>)\s*")


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", ARROW.sub(" thành ", t or "")).strip()


def rationale_texts(sample: dict, v: str, reply: dict) -> dict[int, str] | None:
    """{step index: "<select>...</select>"} or None when any step failed the code check."""
    res = R.check(sample, R.steps_for(sample, v), reply.get("content", ""), v)
    if any(r["bad"] for r in res):
        return None
    out = {}
    for r, c in zip(res, sample["label"]["tool_calls"]):
        args = f" <args>{clean(str(r['args']))}</args>" if r["args"] and not TOOL_ONLY else ""
        out[r["i"]] = f"<select>{clean(r['select'])} <tool>{c['name']}</tool>{args}</select>"
    return out


def row_for(sample: dict, v: str, texts: dict[int, str] | None, split: str) -> dict:
    if v == "gold":
        p = build_prompt(sample, "train", sample["audio_text"], None, normalise=False)
    else:
        p = build_prompt(sample, "train", R.asr_text()[sample["audio"]], R.asr_text(), normalise=False)
    return {"id": sample["id"], "variant": v, "split": split, "system": p["system"], "tools": p["tools"],
            "messages": p["messages"] + trained_turns(sample, texts, tool_kinds())}


def recovery_row(row: dict, rng: random.Random) -> dict | None:
    """recovery_data.make on a RAT row; the lookup after the guard error gets a first rationale sentence about it."""
    got = make_recovery(row, row["messages"], rng)
    if not got:
        return None
    msgs = got["messages"]
    k = next(i for i, m in enumerate(msgs) if m.get("train") is False)  # the blocked call
    blocked, lookup = msgs[k], msgs[k + 2]
    name = blocked["tool_calls"][0]["function"]["name"]
    fakes = ", ".join(f for _, f in got["fakes"])
    # the blocked call: what the model would have written before guessing (no loss, context only)
    sel = re.match(r"<select>(.*?) <tool>", next(m for m in msgs[k + 2:] if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == name)["content"] or "")
    blocked["content"] = f"<select>{sel.group(1) if sel else ''} <tool>{name}</tool></select>"
    lookup["content"] = lookup["content"].replace(
        "<select>", f"<select>Lời gọi trước bị chặn vì mã {fakes} chưa thấy trong câu nói hay kết quả nào; phải tra cứu mã thật. ", 1)
    return {**row, "id": f"{row['id']}_recovery", "variant": f"{row['variant']}_recovery", "source_id": row["id"],
            "messages": msgs, "recovery": {"call": got["call"], "lookup": got["lookup"], "fakes": got["fakes"]}}


def show(row: dict) -> None:
    msgs = row["messages"]
    print(f"\n== {row['id']}  ({row['variant']})\n   {msgs[body_start(msgs) - 1]['content']}")
    for k in range(body_start(msgs), len(msgs)):
        m = msgs[k]
        if m["role"] == "assistant":
            if m.get("content"):
                print(f"    {m['content']}" + ("   [KHÔNG TRAIN]" if m.get("train") is False else ""))
            for c in m.get("tool_calls") or []:
                print(f"    -> {c['function']['name']} {c['function']['arguments']}")
        elif m["role"] == "tool":
            print(f"         = {m['content'][:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--tool-only", action="store_true", help="omit parameter-reasoning spans; train only tool selection reasoning")
    a = ap.parse_args()
    global TOOL_ONLY, OUT
    TOOL_ONLY = a.tool_only
    if TOOL_ONLY:
        OUT = PROJECT / "work" / "sft_rat_toolonly"
    rng = random.Random(a.seed)
    samples = [s for s in load_samples("train") if R.CV.get(s["id"]) not in (0, None)]
    split = splits(load_samples("train"))
    replies = {v: R.load(v) for v in ("gold", "asr")}
    rows, drop = {"train": [], "val": []}, Counter()
    for s in samples:
        sp = split[s["id"]]
        if not s["label"]["tool_calls"]:
            for v in (("gold",) if sp == "val" else ("gold", "asr")):
                rows[sp].append(row_for(s, v, None, sp))
            continue
        for v in (("gold",) if sp == "val" else ("gold", "asr")):
            reply = replies[v].get(s["id"])
            texts = rationale_texts(s, v, reply) if reply else None
            if texts is None:
                drop[f"{v}: {'no reply' if not reply else 'failed check'}"] += 1
                continue
            rows[sp].append(row_for(s, v, texts, sp))

    # recovery rows from the clean train rows, round-robin over the app of the repaired call
    cands = defaultdict(list)
    for r in rows["train"]:
        if r["variant"] in ("gold", "asr") and len(r["messages"]) > 1:
            got = recovery_row(json.loads(json.dumps(r)), rng)
            if got:
                cands[got["recovery"]["call"].split(".")[0]].append(got)
    for app in cands:
        rng.shuffle(cands[app])
    picked, used = [], set()
    while len(picked) < a.recovery and any(cands.values()):
        for app in sorted(cands, key=lambda x: -len(cands[x])):
            while cands[app] and cands[app][-1]["source_id"] in used:
                cands[app].pop()
            if cands[app] and len(picked) < a.recovery:
                r = cands[app].pop()
                used.add(r["source_id"])
                picked.append(r)

    gold = build_gold("train")
    by_id = {s["id"]: s for s in samples}
    bad = {r["id"]: e for r in picked if (e := recovery_check(r, by_id[r["source_id"]], gold[r["source_id"]]["state_hash"]))}
    if bad:
        raise SystemExit(f"{len(bad)} recovery rows fail their check, e.g. {next(iter(bad.items()))}")
    for r in rows["train"] + rows["val"]:  # plain rows: the trained calls replay to the gold hash
        calls = [json.loads(json.dumps(c)) for m in r["messages"][body_start(r["messages"]):] for c in (m.get("tool_calls") or [])]
        if [c["function"]["name"] for c in calls] != [c["name"] for c in by_id[r["id"]]["label"]["tool_calls"]]:
            raise SystemExit(f"{r['id']} {r['variant']}: trained calls differ from gold")
    rows["train"] += picked

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    stats = {}
    for sp in rows:
        enc = []
        for r in rows[sp]:
            try:
                enc.append(encode(r, tok, MAX_LEN))
            except ValueError as e:
                raise SystemExit(str(e))
        stats[sp] = enc
    if any(R.CV.get(r.get("source_id", r["id"])) == 0 for sp in rows for r in rows[sp]):
        raise SystemExit("a row comes from CV fold 0")
    if {r["id"] for r in rows["val"]} & {r.get("source_id", r["id"]) for r in rows["train"]}:
        raise SystemExit("val samples leak into train")

    OUT.mkdir(parents=True, exist_ok=True)
    for sp in rows:
        rng.shuffle(rows[sp])
        (OUT / f"traj.{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[sp]), encoding="utf-8")
    for sp, enc in stats.items():
        n = [e["n_tokens"] for e in enc]
        sup = [e["n_supervised"] for e in enc]
        kinds = Counter(r["variant"] for r in rows[sp])
        print(f"{sp}: {len(rows[sp])} rows {dict(kinds)} | tokens median {st.median(n):.0f}, max {max(n)} | "
              f"supervised median {st.median(sup):.0f}, max {max(sup)}")
    print("dropped:", dict(drop))
    sel_tok = [len(tok(m["content"], add_special_tokens=False)["input_ids"]) for r in rows["train"] for m in r["messages"]
               if m["role"] == "assistant" and m.get("train") and m.get("tool_calls") and m.get("content")]
    print(f"rationale tokens per step: median {st.median(sel_tok):.0f}, p90 {sorted(sel_tok)[int(.9 * len(sel_tok))]}, max {max(sel_tok)}")
    print(f"-> {OUT}")
    shown = [r for r in rows["train"] if r["variant"].endswith("recovery")][:1] + [r for r in rows["train"] if r["variant"] == "asr"][:max(0, a.show - 1)]
    for r in shown[:a.show]:
        show(r)


if __name__ == "__main__":
    main()
