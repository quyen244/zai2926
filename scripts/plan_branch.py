"""Conditional labels for the experiment-A plans (docs/plan_format.md, rule R3): which gold call checks the condition.

  python scripts/plan_branch.py --show train_0006 train_0001     # prompt input + rule draft + the resulting plans (screen)
  python scripts/plan_branch.py --show 6 --system                # also the system prompt
  python scripts/plan_branch.py --requests                       # write work/sft/plan_check.requests.jsonl for the job
  python scripts/plan_branch.py --merge work/kaggle/plan_check/labels.jsonl   # validate -> work/sft/conditional_labels.jsonl

The Kaggle job (kaggle/plan_check/, Qwen3-8B-AWQ on vLLM) sends exactly the prompts of --requests: system =
prompts/plan_branch.system.md, user = prompts/plan_branch.user.md with the placeholders filled (HTML comments stripped),
JSON schema {"conditional": bool, "check_call": int|null}. --merge checks every reply: check_call must be a call index
of the trajectory and a READ tool; a reply that fails is kept as not conditional (valid = false) and counted.
"Rule draft" in --show is NOT the model's answer: regex tag "conditional" (work/tags.jsonl) + the first read call; it
is only there so the prompt can be reviewed against something. Per-sample output stays under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from lifemate.prompt import build_tools  # noqa: E402
from plan_data import ASK, plan_segments  # noqa: E402

PROMPTS = PROJECT / "prompts"
REQUESTS = PROJECT / "work" / "sft" / "plan_check.requests.jsonl"
LABELS = PROJECT / "work" / "sft" / "conditional_labels.jsonl"
# Tier 1 (docs/plan_format.md, decided 2026-11-08): a request with no condition word is never R3, whatever the model
# says; the first pass called 745 samples conditional, about half of them plain lookups.
MARKER = re.compile(r"\b(thì|nếu|hễ|lỡ|trường hợp|miễn là|còn không|không thì|kẻo)\b", re.I)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["conditional", "check_call"],
          "properties": {"conditional": {"type": "boolean"}, "check_call": {"type": ["integer", "null"]}}}


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


def system_prompt() -> str:
    return strip_comments((PROMPTS / "plan_branch.system.md").read_text(encoding="utf-8"))


def render_user(s: dict, kinds: dict[str, str]) -> str:
    hist = []
    for h in s["history"]:
        if h["role"] == "user":
            hist.append(f"user: {h.get('audio_text', '')}")
        else:
            for c, r in zip(h.get("tool_calls") or [], h.get("tool_results") or []):
                hist.append(f"assistant gọi {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) -> {json.dumps(r, ensure_ascii=False)[:300]}")
            if h.get("text"):
                hist.append(f"assistant: {h['text']}")
    tools = build_tools("train", s["context"]["installed_apps"])
    inst = "\n".join(f"{t['function']['name']} [{'ask' if t['function']['name'] == ASK else kinds.get(t['function']['name'], 'read')}]: {t['function']['description']}" for t in tools)
    calls, res = s["label"]["tool_calls"], s["label"]["tool_results"]
    traj = "\n".join(f"{k} {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)} -> {json.dumps(r, ensure_ascii=False)[:300]}" for k, (c, r) in enumerate(zip(calls, res), 1))
    fill = {"current_time": s["context"]["current_time"], "history": "\n".join(hist) or "(không có)", "request": s["audio_text"],
            "installed_tools": inst, "trajectory": traj or "(không có)", "n_calls": str(len(calls))}
    text = strip_comments((PROMPTS / "plan_branch.user.md").read_text(encoding="utf-8"))
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def rule_draft(s: dict, kinds: dict[str, str], tags: dict[str, set]) -> dict:
    calls = s["label"]["tool_calls"]
    reads = [k for k, c in enumerate(calls) if kinds.get(c["name"]) == "read"]
    if "conditional" not in tags.get(s["id"], set()) or not reads:
        return {"conditional": False, "check_call": None}
    return {"conditional": True, "check_call": reads[0] + 1}


def validate(s: dict, reply: dict, kinds: dict[str, str]) -> tuple[bool, str]:
    if not isinstance(reply, dict) or not isinstance(reply.get("conditional"), bool):
        return False, "bad json"
    if not reply["conditional"]:
        return reply.get("check_call") is None, "ok" if reply.get("check_call") is None else "check_call set but not conditional"
    k = reply.get("check_call")
    calls = s["label"]["tool_calls"]
    if not isinstance(k, int) or not 1 <= k <= len(calls):
        return False, "check_call out of range"
    if kinds.get(calls[k - 1]["name"]) != "read":
        return False, "check_call is not a read tool"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", help="sample ids (a bare number works): prompt input, rule draft, plans")
    ap.add_argument("--system", action="store_true", help="also print the system prompt")
    ap.add_argument("--requests", action="store_true", help=f"write {REQUESTS.relative_to(PROJECT)} (one request per train sample)")
    ap.add_argument("--merge", type=Path, help="validate the job's labels.jsonl and write the labels file plan_data.py reads")
    a = ap.parse_args()

    samples = {s["id"]: s for s in load_samples("train")}
    kinds = {t["name"]: t["kind"] for x in load_catalog("train")["apps"] for t in x["tools"]}
    tags = {(r := json.loads(l))["id"]: set(r["linguistic"]) | set(r["structural"]) for l in (PROJECT / "work/tags.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    if a.system:
        print("=" * 100 + "\nSYSTEM PROMPT (prompts/plan_branch.system.md)\n" + "=" * 100 + "\n" + system_prompt())
    if a.requests:
        REQUESTS.parent.mkdir(parents=True, exist_ok=True)
        sysp = system_prompt()
        with open(REQUESTS, "w", encoding="utf-8") as f:
            for s in samples.values():
                f.write(json.dumps({"id": s["id"], "system": sysp, "user": render_user(s, kinds), "schema": SCHEMA}, ensure_ascii=False) + "\n")
        print(f"-> {REQUESTS} ({len(samples)} requests)")
    if a.merge:
        got = {(r := json.loads(l))["id"]: r for l in a.merge.read_text(encoding="utf-8").splitlines() if l.strip()}
        why, out = Counter(), []
        for i, s in samples.items():
            if i not in got:
                why["missing"] += 1
                continue
            reply = got[i].get("reply")
            ok, reason = validate(s, reply, kinds)
            if ok and reply["conditional"] and not MARKER.search(s["audio_text"]):
                reply, reason = {"conditional": False, "check_call": None}, "tier 1: no condition word"
            why[reason] += 1
            out.append({"id": i, "valid": ok, "reason": reason, **({"conditional": reply["conditional"], "check_call": reply.get("check_call")} if ok else {"conditional": False, "check_call": None})})
        LABELS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        cond = [r for r in out if r["conditional"]]
        reg = {i for i, s in samples.items() if rule_draft(s, kinds, tags)["conditional"]}
        print(f"replies: {len(got)} of {len(samples)} | checks: {dict(why)}")
        print(f"conditional (model, valid): {len(cond)} | regex candidates: {len(reg)} | both: {len({r['id'] for r in cond} & reg)} | "
              f"model only: {len({r['id'] for r in cond} - reg)} | regex only: {len(reg - {r['id'] for r in cond})}")
        print(f"-> {LABELS}")
    for raw in a.show or []:
        sid = raw if raw.startswith("train_") else f"train_{int(raw):04d}"
        s = samples.get(sid)
        if s is None:
            print(f"\n{sid}: not a train id")
            continue
        lab = rule_draft(s, kinds, tags)
        print("\n" + "=" * 100 + f"\n{sid}  INPUT (user prompt sent to Qwen3-8B-AWQ)\n" + "=" * 100 + "\n" + render_user(s, kinds))
        print("-" * 100 + "\nRULE DRAFT of the expected reply (the model decides; this is only for review)\n" + "-" * 100)
        print(json.dumps(lab, ensure_ascii=False))
        print("-" * 100 + "\nPLANS experiment A would learn with that label (assistant message index: plan)\n" + "-" * 100)
        for m, t in plan_segments(s, lab["check_call"]).items():
            print(f"   message {m}: {t}")


if __name__ == "__main__":
    main()
