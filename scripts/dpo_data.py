"""DPO pairs from RAT's own greedy rollouts on train folds 1-4 (never CV fold 0): same prefix, chosen = the gold step,
rejected = the step RAT actually took there.

For each rolled-out sample whose state hash is wrong, walk RAT's replies against the gold trajectory
(scripts/plan_data.trained_turns: one call per assistant message, then a fixed final text):
  - a reply whose single call equals the gold call exactly (name and arguments) -> same tool result, keep walking;
  - the first reply that differs is the divergence: prefix = prompt + RAT's actual prior replies and matching tool results,
    chosen = the gold step with its tool-selection rationale, rejected = RAT's reply there. Kinds: wrong_call (a different call), stop_early (text where gold still calls),
    extra_call (a call where gold has finished).
Skipped: a divergent call that still matches gold under the scoring rule (fuzzy argument, lifemate.scoring.calls_match:
not wrong, and its tool result could differ), a reply with unparsable arguments, no divergence found.
The prompt is rebuilt with build_prompt(normalise=False), matching RAT training, and the full installed tool list.
Rollout cache rows carry their installed app names in keep_apps so the exact offered tools can be reconstructed.
Split: the 60 fold-1 samples of plan_data's val split -> val, the rest -> train.

  python scripts/dpo_data.py --rollouts <gold>.jsonl:gold <asr>.jsonl:asr_ft --normalise false --tag rat
Output: work/dpo/<tag>.{train,val}.jsonl, rows {"id", "variant", "split", "kind", "step", "system", "tools", "messages", "chosen", "rejected"}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import copy
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, fuzzy_params, load_catalog, load_samples  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from lifemate.scoring import calls_match  # noqa: E402
from plan_data import splits, trained_turns  # noqa: E402

OUT = PROJECT / "work" / "dpo"
E5 = PROJECT / "work/kaggle/embed_train_q/out/app_embedder_e5"
ASR_DIR = {"asr_ft": "phowhisper_ft"}
MAX_LEN = 10240


def reply_msg(r: dict) -> dict | None:
    """RAT's reply as a chat message (arguments kept as the JSON string it produced); None if an argument is not JSON."""
    calls = []
    for c in r.get("tool_calls") or []:
        try:
            json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return None
        calls.append({"id": c.get("id") or "x", "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"] or "{}"}})
    m = {"role": "assistant", "content": r.get("content")}
    if calls:
        m["tool_calls"] = calls
    return m


def as_call(m: dict) -> dict:
    f = m["tool_calls"][0]["function"]
    return {"name": f["name"], "arguments": json.loads(f["arguments"] or "{}")}


def rat_gold_steps(row: dict | None) -> list[dict]:
    if row is None:
        return []
    return [m for m in row["messages"] if m.get("role") == "assistant" and m.get("tool_calls") and m.get("train")]


def rollout_prefix(prompt_messages: list[dict], gold_msgs: list[dict], replies: list[dict], n_calls: int) -> list[dict]:
    """Use RAT's observed prior replies and the matching tool results as the shared DPO prefix."""
    prefix = copy.deepcopy(prompt_messages)
    call_steps = [i for i, m in enumerate(gold_msgs) if m.get("role") == "assistant" and m.get("tool_calls")]
    tool_steps = [i for i, m in enumerate(gold_msgs) if m.get("role") == "tool"]
    for k in range(n_calls):
        reply = reply_msg(replies[k])
        if reply is None or not reply.get("tool_calls"):
            raise ValueError(f"missing call at rollout prefix step {k}")
        prefix.append(reply)
        result = copy.deepcopy(gold_msgs[tool_steps[k]])
        result["tool_call_id"] = reply["tool_calls"][0].get("id") or "x"
        prefix.append(result)
    return prefix


def diverge(gold_msgs: list[dict], replies: list[dict], fz) -> tuple[str, int, dict, dict] | str:
    """(kind, step, chosen, rejected) at the first step where RAT left the gold path, or a skip reason."""
    steps = [m for m in gold_msgs if m["role"] == "assistant"]
    for k, g in enumerate(steps):
        if k >= len(replies):
            return "skip_no_reply"
        r = reply_msg(replies[k])
        if r is None:
            return "skip_bad_json"
        chosen = {kk: v for kk, v in g.items() if kk != "train"}
        if not g.get("tool_calls"):  # gold has finished
            return ("extra_call", k, chosen, r) if r.get("tool_calls") else "skip_no_divergence"
        if not r.get("tool_calls"):
            return "stop_early", k, chosen, r
        gc, rc = as_call(g), as_call(r)
        if len(r["tool_calls"]) == 1 and gc == rc:
            continue
        if len(r["tool_calls"]) == 1 and calls_match(rc, gc, fz):
            return "skip_fuzzy_equal"
        return "wrong_call", k, chosen, r
    return "skip_no_divergence"


def show(row: dict) -> None:
    def line(m: dict) -> str:
        if m.get("tool_calls"):
            return "\n".join(f"    -> {c['function']['name']} {c['function']['arguments']}" for c in m["tool_calls"])
        return f"    (text) {m.get('content')}"
    user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    print(f"== {row['id']} [{row['variant']}] {row['kind']} at step {row['step']}")
    print(f"  user: {user}")
    body = row["messages"][max(k for k, m in enumerate(row["messages"]) if m["role"] == "user") + 1:]
    for m in body:
        if m["role"] == "assistant":
            print(line(m).replace("    ->", "    .. prefix ->"))
        elif m["role"] == "tool":
            print(f"         = {m['content'][:160]}")
    print("  chosen:")
    print(line(row["chosen"]))
    print("  rejected:")
    print(line(row["rejected"]))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True, help="cache.jsonl:variant (variant gold or asr_ft)")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--tag", default="rec")
    ap.add_argument("--normalise", choices=["true", "false"], default="true")
    a = ap.parse_args()

    cv = json.loads((PROJECT / "splits/cv5.json").read_text(encoding="utf-8"))
    samples = {s["id"]: s for s in load_samples("train")}
    sp = splits(list(samples.values()))
    kinds = {t["name"]: t["kind"] for x in load_catalog("train")["apps"] for t in x["tools"]}
    gold = build_gold("train")
    rat_rows = {}
    rat_dir = PROJECT / "work/sft_rat_toolonly"
    for part in ("train", "val"):
        for line in (rat_dir / f"traj.{part}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rr = json.loads(line)
            if rr.get("source_id"):
                continue  # recovery augmentation is not a canonical sample trajectory
            v = "asr_ft" if rr["variant"] == "asr" else rr["variant"]
            rat_rows[(rr["id"], v)] = rr
    fz = fuzzy_params("train")
    asr = {v: {(j := json.loads(l))["audio"]: j["hyp"] for l in (PROJECT / "work/asr" / d / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
           for v, d in ASR_DIR.items()}
    filt = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    from lifemate.sft import encode

    rows, stats = [], Counter()
    for spec in a.rollouts:
        path, variant = spec.rsplit(":", 1)
        for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()):
            sid = r["id"]
            if cv[sid] == 0:
                raise SystemExit(f"{sid} is CV fold 0: rollouts must come from folds 1-4")
            stats[f"{variant}.samples"] += 1
            if r["state_hash"] == gold[sid]["state_hash"]:
                stats[f"{variant}.hash_ok"] += 1
                continue
            s = samples[sid]
            text = s["audio_text"] if variant == "gold" else asr[variant][s["audio"]]
            keep = set(r["keep_apps"]) if "keep_apps" in r else None
            if keep is None:
                if filt is None:
                    from lifemate.app_filter import AppFilter
                    filt = AppFilter(E5, device="cpu")
                keep = filt.keep(text, list(s["context"]["installed_apps"]))
            p = build_prompt(s, "train", text, asr.get(variant), normalise=a.normalise == "true")
            tools = [t for t in p["tools"] if t["function"]["name"].split(".")[0] in keep | {"sys"}]
            rationale_row = rat_rows.get((sid, variant))
            if rationale_row is None:
                stats[f"{variant}.skip_no_rationale"] += 1
                continue
            gmsgs = trained_turns(s, None, kinds)
            rationale_steps = rat_gold_steps(rationale_row)
            gold_steps = [m for m in gmsgs if m.get("role") == "assistant" and m.get("tool_calls")]
            if len(rationale_steps) != len(gold_steps):
                stats[f"{variant}.skip_rationale_step_mismatch"] += 1
                continue
            for gm, rm in zip(gold_steps, rationale_steps):
                gm["content"] = rm.get("content")
            d = diverge(gmsgs, r["replies"], fz)
            if isinstance(d, str):
                stats[f"{variant}.{d}"] += 1
                continue
            kind, k, chosen, rejected = d
            # prefix = prompt + gold messages up to (not including) the k-th assistant message
            idx = [i for i, m in enumerate(gmsgs) if m["role"] == "assistant"][k]
            n_prefix_calls = sum(1 for m in gmsgs[:idx] if m.get("role") == "assistant" and m.get("tool_calls"))
            prefix = rollout_prefix(p["messages"], gmsgs, r["replies"], n_prefix_calls)
            row = {"id": sid, "variant": variant, "split": "val" if sp[sid] == "val" else "train", "kind": kind, "step": k,
                   "system": p["system"], "tools": tools, "messages": p["messages"] + prefix, "chosen": chosen, "rejected": rejected}
            n = max(encode({**row, "messages": row["messages"] + [{**m, "train": True}]}, tok, 10 ** 6)["n_tokens"] for m in (chosen, rejected))
            if n > MAX_LEN:
                stats[f"{variant}.skip_too_long"] += 1
                continue
            stats[f"{variant}.{kind}"] += 1
            rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val"):
        rs = [r for r in rows if r["split"] == part]
        (OUT / f"{a.tag}.{part}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        print(f"{a.tag}.{part}: {len(rs)} pairs ({len({r['id'] for r in rs})} samples)")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    for r in random.Random(0).sample(rows, min(a.show, len(rows))):
        show(r)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
