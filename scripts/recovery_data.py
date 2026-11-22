"""Recovery rows for the ID guard (plan item 2): SFT rows that teach the model what to do after the guard blocks a call
with an invented ID: look the thing up, then redo the call with the real ID. Train folds only (never CV fold 0).

Source: the B2 training rows (work/sft_b2/traj.train.jsonl, paraphrased utterances, filtered tool lists). A source row
qualifies when it has at most 4 tool calls and a write call whose ID argument comes from an earlier lookup result
(not from the prompt). The row is copied and, right before that lookup, two messages are inserted:
  assistant: the same write call with the ID replaced by an invented one in B2's own styles   (train: false, no loss)
  tool:      the guard's error, lifemate.agent.guard_error()
Everything else stays: the lookup, the write with the real ID, the remaining steps and the final reply, all trained.
The utterance is one of the row's paraphrases or the gold text (prompt rebuilt with build_prompt), chosen at random.
One row per source sample, spread round-robin across apps, N rows (default 300).

Checks per row: the trained calls replayed on a fresh ToolEnv give the gold state hash and the stored tool results; a
simulated guard over the row blocks exactly the inserted call and nothing else.

  python scripts/recovery_data.py                 # writes work/recovery/recovery.train.jsonl, prints aggregates
  python scripts/recovery_data.py --show 3        # also prints 3 rows in the show_gt layout
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.agent import guard_error, unseen_ids  # noqa: E402
from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.gold import replay  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402

SRC = PROJECT / "work/sft_b2/traj.train.jsonl"
OUT = PROJECT / "work/recovery/recovery.train.jsonl"
CV = json.loads((PROJECT / "splits" / "cv5.json").read_text(encoding="utf-8"))
MAX_CALLS = 4  # + the blocked call = 5 replies with calls, under the step cap of 6


def body_start(msgs: list[dict]) -> int:
    """Index of the first message after the final user turn (the agent's own steps)."""
    return max(k for k, m in enumerate(msgs) if m["role"] == "user") + 1


def calls_of(msgs: list[dict]) -> list[tuple[int, str, dict]]:
    """The agent's own calls of the current turn (history turns may hold calls too; those are prompt)."""
    out = []
    for k, m in enumerate(msgs):
        for c in (m.get("tool_calls") or []) if k >= body_start(msgs) else []:
            a = c["function"]["arguments"]
            out.append((k, c["function"]["name"], a if isinstance(a, dict) else json.loads(a or "{}")))
    return out


def fake_id(v: str, rng: random.Random, avoid: str) -> str:
    """An invented ID in the styles B2 made up on fold 0 (rem_123456, contact_123, beneficiary_01, 1234567890)."""
    m = re.match(r"^(.*?)(\d+)$", v)
    for _ in range(100):
        if m and m.group(1):
            p = m.group(1)
            f = rng.choice([p + str(rng.randint(100, 999)), p + f"{rng.randint(1, 999999):06d}", p + f"{rng.randint(1, 9):02d}"])
        elif m:
            f = "".join(rng.choice("0123456789") for _ in v)
        else:
            f = v.rsplit("_", 1)[0] + f"_{rng.randint(1, 999)}"
        if f != v and f not in avoid:
            return f
    raise RuntimeError(f"no fake ID for {v}")


def seen_before(system: str, msgs: list[dict]) -> dict[int, str]:
    """For each message index of an assistant call: what the guard has seen before it (prompt + earlier results)."""
    start = body_start(msgs)
    seen = json.dumps([system, msgs[:start]], ensure_ascii=False)
    out = {}
    for k in range(start, len(msgs)):
        if msgs[k]["role"] == "assistant":
            out[k] = seen
        elif msgs[k]["role"] == "tool":
            seen += msgs[k]["content"]
    return out


def make(row: dict, msgs: list[dict], rng: random.Random) -> dict | None:
    calls = calls_of(msgs)
    if not calls or len(calls) > MAX_CALLS:
        return None
    seen = seen_before(row["system"], msgs)
    prompt_seen = seen[calls[0][0]]
    for k, name, args in calls:
        if name.startswith("sys."):
            continue
        from_result = [v for v in unseen_ids(args, prompt_seen) if v in seen[k]]
        if not from_result:
            continue
        v = from_result[0]
        src = next(kk for kk, _, _ in calls if kk < k and v in msgs[kk + 1]["content"])  # the lookup that returned v
        avoid = json.dumps(msgs, ensure_ascii=False)
        bad = dict(args)
        fakes = []
        for key, val in args.items():  # every ID the guard would not know yet at the lookup gets invented
            if val in unseen_ids({key: val}, seen[src]):
                bad[key] = fake_id(val, rng, avoid)
                fakes.append((val, bad[key]))
        blocked = unseen_ids(bad, seen[src])
        if not blocked:
            return None
        cid = f"guard_{src}"
        inserted = [{"role": "assistant", "content": None, "train": False, "tool_calls": [{
                        "id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(bad, ensure_ascii=False)}}]},
                    {"role": "tool", "tool_call_id": cid, "content": json.dumps(guard_error(blocked), ensure_ascii=False)}]
        return {"messages": msgs[:src] + inserted + msgs[src:], "call": name, "fakes": fakes,
                "lookup": next(n for kk, n, _ in calls if kk == src)}
    return None


def check(row: dict, sample: dict, gold_hash: str) -> list[str]:
    msgs = row["messages"]
    errs = []
    seen = seen_before(row["system"], msgs)
    trained, results = [], []
    for k, name, args in calls_of(msgs):
        blocked = [] if name.startswith("sys.") else unseen_ids(args, seen[k])
        if bool(blocked) != (msgs[k].get("train") is False):
            errs.append(f"guard mismatch at message {k}: blocked {blocked}, train {msgs[k].get('train')}")
        if msgs[k].get("train"):
            trained.append({"name": name, "arguments": args})
            results.append(json.loads(msgs[k + 1]["content"]))
    env, got = replay("train", sample["id"], trained)
    if env.get_state_hash() != gold_hash:
        errs.append("state hash differs from gold")
    if got != results:
        errs.append("replayed results differ from the row's tool messages")
    return errs


def show(row: dict) -> None:
    msgs = row["messages"]
    user = msgs[body_start(msgs) - 1]["content"]
    print(f"== {row['id']}  ({row['variant']}, fold {CV.get(row['source_id'])})  sửa: {row['recovery']['call']}")
    n_hist = sum(len(m.get("tool_calls") or []) for m in msgs[:body_start(msgs)])
    print(f"   {user}" + (f"\n   (lịch sử hội thoại có {n_hist} lời gọi, ẩn)" if n_hist else ""))
    for k, m in enumerate(msgs):
        for c in (m.get("tool_calls") or []) if k >= body_start(msgs) else []:
            tag = "   [KHÔNG TRAIN, bị chặn]" if m.get("train") is False else ""
            print(f"    -> {c['function']['name']} {c['function']['arguments']}{tag}")
            print(f"         = {msgs[k + 1]['content'][:200]}")
        if m["role"] == "assistant" and not m.get("tool_calls") and k >= body_start(msgs):
            print(f"    <- {m['content']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print this many rows in the show_gt layout")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    samples = {s["id"]: s for s in load_samples("train")}
    by_id = defaultdict(list)
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[r["id"]].append(r)
    if any(CV.get(i) == 0 for i in by_id):
        raise SystemExit("source rows contain CV fold 0 samples")

    cands = defaultdict(list)  # app of the repaired call -> built rows
    for sid in sorted(by_id):
        rows = by_id[sid]
        sample = samples[sid]
        variant = rng.choice(["gold"] + [r["variant"] for r in rows])
        base = next(r for r in rows if r["variant"] == variant) if variant != "gold" else rows[0]
        msgs = base["messages"]
        if variant == "gold":
            built = build_prompt(sample, "train", sample["audio_text"])
            if built["system"] != base["system"]:
                raise SystemExit(f"{sid}: build_prompt system differs from the B2 row's")
            msgs = built["messages"] + msgs[body_start(msgs):]
        got = make(base, msgs, rng)
        if got:
            cands[got["call"].split(".")[0]].append({
                "id": f"{sid}_recovery", "source_id": sid, "variant": variant, "split": "train",
                "system": base["system"], "tools": base["tools"], "messages": got["messages"],
                "recovery": {"call": got["call"], "lookup": got["lookup"], "fakes": got["fakes"]}})
    for v in cands.values():
        rng.shuffle(v)
    picked, apps = [], sorted(cands, key=lambda x: -len(cands[x]))
    while len(picked) < a.n and any(cands.values()):
        for app in apps:
            if cands[app] and len(picked) < a.n:
                picked.append(cands[app].pop())

    from lifemate.gold import build_gold
    gold = build_gold("train")
    bad = {r["id"]: e for r in picked if (e := check(r, samples[r["source_id"]], gold[r["source_id"]]["state_hash"]))}
    total = sum(len(v) for v in cands.values()) + len(picked)
    print(f"source samples {len(by_id)}, qualifying {total}, picked {len(picked)}, failed checks {len(bad)}")
    for i, e in list(bad.items())[:5]:
        print(f"  FAIL {i}: {e}")
    print("variants:", dict(Counter(r["variant"] for r in picked)))
    print("apps:", ", ".join(f"{k} {c}" for k, c in Counter(r["recovery"]["call"].split(".")[0] for r in picked).most_common()))
    print("repaired calls:", ", ".join(f"{k} {c}" for k, c in Counter(r["recovery"]["call"] for r in picked).most_common(8)))
    if bad:
        raise SystemExit("rows failed checks; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    print(f"-> {OUT}\n")
    for r in rng.sample(picked, min(a.show, len(picked))):
        show(r)


if __name__ == "__main__":
    main()
