"""Build the two SFT datasets of the plan ablation from the gold trajectories (no LLM): A = plan + trajectory, B = trajectory only.

  python scripts/plan_data.py --show train_0921 train_0001     # print the plan and the A / B assistant turns (screen only)
  python scripts/plan_data.py                                  # build + validate -> work/sft/{plan,traj}.{train,val,test}.jsonl
  python scripts/plan_data.py --para work/synth/paraphrase.kept.jsonl --out work/sft_para   # + one train row per kept rewrite

Same tasks, same calls, same tool results in A and B; the only difference is the plan text in the content of the first
assistant message of the final turn (A): the gold tool chain "<plan>contacts.search -> phone.call -> end</plan>"
("<plan>end</plan>" when the gold calls nothing). make_plan() below also keeps a richer structure (goal, kinds,
inputs, depends_on, expected_output) that is NOT trained; scripts/ablation_metrics.py uses it to define the
data-dependent category. It is derived from the gold label and the tool catalog only:
  goal             template from the call kinds: "Không cần gọi tool" / "Tra cứu: ..." / "Tra cứu ... rồi thực hiện: ..."
                   (+ "rồi hỏi lại <params>" when the gold asks through sys.ask_missing_info)
  steps            one per gold call, in order: id s<k>, tool, subtask (catalog description, first clause), kind
                   (read/write/ask), inputs {param: "s<j>"} for argument values taken from an earlier result,
                   depends_on (the s<j> of those inputs), expected_output (catalog "returns")
  dependency rule  an argument value depends on call j when it appears as a value in call j's RESULT, and is not in the
                   user's text, the context/profile, or any earlier call's ARGUMENTS (then it came from the user)
No value from a tool result is written in the plan (inputs name the step, not the value).
Final assistant message (A and B alike, only the stop signal matters for the score): fixed short text per outcome.
Rows: {"id", "variant": "gold"|"asr", "split", "system", "tools", "messages"}; messages = the agent prompt (lifemate.prompt,
history replayed) + the trained turns, each trained assistant message marked "train": true (history turns are not trained).
Splits (identical for A and B): test = cv5 fold 0 (309), val = 60 samples of fold 1 (seed 0), train = the rest.
Training rows carry both text variants (reference transcript and cached PhoWhisper ASR); val/test rows are gold text.
Validation before writing: every gold call is replayed on a fresh ToolEnv and must return the label's tool_results and
reach the label's state hash; every plan must map its steps 1:1 onto the gold calls; every row must fit 10240 tokens.
Per-sample data: everything is written under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_catalog, load_samples, tool_env  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.prompt import build_prompt, build_user_text  # noqa: E402
from lifemate.sft import encode  # noqa: E402

OUT = PROJECT / "work" / "sft"
MAX_LEN = 10240
FINAL = {"action": "Dạ, em đã làm xong.", "read": "Dạ, em đã tra cứu xong.", "ask": "Dạ, anh/chị cho em xin thêm thông tin còn thiếu nhé.",
         "empty": "Dạ, yêu cầu này em không cần dùng app nào."}


def scalars(x) -> list:
    """Every scalar value inside a JSON value (dict values, list items), no keys."""
    if isinstance(x, dict):
        return [v for y in x.values() for v in scalars(y)]
    if isinstance(x, list):
        return [v for y in x for v in scalars(y)]
    return [x]


def meaningful(v) -> bool:
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        return abs(v) >= 10  # tiny numbers (1, 2, 0) collide by chance
    return isinstance(v, str) and len(v.strip()) >= 2


_ENUM: set[tuple[str, str]] | None = None


def enum_params() -> set[tuple[str, str]]:
    """(tool, param) pairs whose schema is an enum (power on/off, cycle, action...): a choice made from the user's words,
    never a value copied from a result, even when the same word appears in one (a device status "off")."""
    global _ENUM
    if _ENUM is None:
        _ENUM = {(t["name"], pn) for x in load_catalog("train")["apps"] for t in x["tools"]
                 for pn, ps in ((t.get("parameters") or {}).get("properties") or {}).items() if "enum" in ps}
    return _ENUM


_KINDS: dict[str, str] | None = None


def tool_kinds() -> dict[str, str]:
    global _KINDS
    if _KINDS is None:
        _KINDS = {t["name"]: t["kind"] for x in load_catalog("train")["apps"] for t in x["tools"]}
    return _KINDS


def dependencies(sample: dict) -> list[dict[str, int]]:
    """Per gold call: {param: index of the earlier call whose result supplied that argument value}."""
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    enums = enum_params()
    # what the model is shown: user turns after the prompt's normaliser (numbers said in words become digits, relative
    # dates get a date hint), not the raw transcript: a value the user said in words is from the user, not a result
    now = sample["context"]["current_time"]
    shown = lambda h: build_user_text(h["audio_text"], now) if h.get("role") == "user" and h.get("audio_text") else h.get("text")  # noqa: E731
    user_side = json.dumps([build_user_text(sample["audio_text"], now), sample["context"], [shown(h) for h in sample["history"]]], ensure_ascii=False)
    out = []
    for j, c in enumerate(calls):
        seen_args = {json.dumps(v, ensure_ascii=False) for cc in calls[:j] for v in scalars(cc["arguments"])}
        dep = {}
        for p, val in c["arguments"].items():
            if (c["name"], p) in enums:
                continue
            for v in scalars(val):
                key = json.dumps(v, ensure_ascii=False)
                if not meaningful(v) or key in seen_args or (isinstance(v, str) and v in user_side) or str(v) in user_side:
                    continue
                src = [i for i in range(j) if key in {json.dumps(x, ensure_ascii=False) for x in scalars(results[i])}]
                if src:
                    dep[p] = src[-1]  # the latest earlier result that holds it
        out.append(dep)
    return out


def depends(sample: dict, deps: list[dict[str, int]]) -> list[list[int]]:
    """depends_on per call: the calls behind its inputs; a run of consecutive calls of the same read tool (e.g. one
    get_activity per day before "the highest day") counts as one aggregate, so the step depends on the whole run."""
    names = [c["name"] for c in sample["label"]["tool_calls"]]
    out = []
    for d in deps:
        src = set()
        for i in d.values():
            lo, hi = i, i
            while lo > 0 and names[lo - 1] == names[i]:
                lo -= 1
            while hi + 1 < len(names) and names[hi + 1] == names[i] and hi + 1 < len(deps) and hi + 1 not in src:
                hi += 1
            src |= set(range(lo, hi + 1))
        out.append(sorted(src))
    return out


def outcome(calls: list[dict], kinds: dict[str, str]) -> str:
    if not calls:
        return "empty"
    if calls[-1]["name"] == "sys.ask_missing_info":
        return "ask"
    return "action" if any(kinds.get(c["name"]) == "write" for c in calls) else "read"


def make_plan(sample: dict, tools: dict[str, dict]) -> dict:
    calls = sample["label"]["tool_calls"]
    deps = dependencies(sample)
    dep_on = depends(sample, deps)
    kind = lambda n: "ask" if n == "sys.ask_missing_info" else tools.get(n, {}).get("kind", "read")  # noqa: E731
    steps = []
    for k, (c, d, on) in enumerate(zip(calls, deps, dep_on), 1):
        t = tools.get(c["name"], {})
        if c["name"] == "sys.ask_missing_info":
            sub = f"hỏi lại {', '.join(c['arguments'].get('params') or [])} cho {c['arguments'].get('tool_name', '')}"
        else:
            sub = t.get("description", "").split(";")[0].split(". ")[0].split(" (")[0].strip()
        st = {"id": f"s{k}", "tool": c["name"], "kind": kind(c["name"]), "subtask": sub}
        if d:
            st["inputs"] = {p: f"s{i + 1}" for p, i in d.items()}
        st["depends_on"] = [f"s{i + 1}" for i in on if i < k - 1]
        st["expected_output"] = (t.get("returns") or "ok") if c["name"] != "sys.ask_missing_info" else "chờ người dùng trả lời"
        steps.append(st)
    reads = [c["name"] for c in calls if kind(c["name"]) == "read"]
    writes = [c["name"] for c in calls if kind(c["name"]) == "write"]
    ask = [c for c in calls if c["name"] == "sys.ask_missing_info"]
    if not calls:
        goal = "Không cần gọi tool"
    else:
        parts = ([f"tra cứu {', '.join(dict.fromkeys(reads))}"] if reads else []) + ([f"thực hiện {', '.join(dict.fromkeys(writes))}"] if writes else [])
        if ask:
            parts.append(f"hỏi lại {', '.join(p for c in ask for p in (c['arguments'].get('params') or []))}")
        goal = (" rồi ".join(parts)).capitalize()
    return {"goal": goal, "steps": steps}


ISO_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}(T|$)")
ASK = "sys.ask_missing_info"


UNKNOWN = "?"


def step_text(name: str, args: dict, dep: dict[str, int], known=None) -> str:
    """tool(arg=value, ...): a value copied from the result of call k is written $k (1-based over the trajectory);
    dates/times are never $k (they are computed, e.g. "one hour later", not copied). known(value) -> bool: for a step
    AFTER the plan's first one, a literal is written only when the model can already see that value when it writes the
    plan; any other value is "?" (it is filled in when the call itself is made, after the results it needs).
    v1 wrote the gold value there, which made the model guess times/names it could not know yet: A lost 21 points to B
    on the 25% of test samples with such a value (2026-11-09, reports/plan_sft_ablation.md)."""
    parts = []
    for p_, v in args.items():
        if p_ in dep and not (isinstance(v, str) and ISO_TIME.match(v)):
            parts.append(f"{p_}=${dep[p_] + 1}")
        elif known is None or (name, p_) in enum_params() or known(v):  # an enum value is one of the schema's (visible)
            parts.append(f"{p_}={json.dumps(v, ensure_ascii=False)}")
        else:
            parts.append(f"{p_}={UNKNOWN}")
    return f"{name}({', '.join(parts)})"


def plan_segments(sample: dict, check_call: int | None) -> dict[int, str]:
    """{assistant message index: plan text} for experiment A (docs/plan_format.md). Message k carries gold call k
    (0-based); message n = the final reply. Rules:
      R1 plain chain: one plan at message 0, the whole chain, "-> end"
      R2 lookup then ask (the lookup found several matches): the plan before the lookup is the happy path, the tool named
         by sys.ask_missing_info.tool_name with its missing params taken from the lookup ($k), "-> end"; the ask message
         carries a new plan "sys.ask_missing_info(...) -> end"
      R3 conditional (check_call from the Qwen3-8B label): the plan stops after the check call with "-> replan"; the
         message right after the check result carries the plan of the rest (or "<plan>end</plan>")
      R4 no call: "<plan>end</plan>" on the final reply
    A plan never holds a value the model could only know after a tool result: after the plan's first step, a literal is
    kept only if it is visible when the plan is written (user text, context, history, earlier calls and results), else "?"."""
    calls = sample["label"]["tool_calls"]
    deps = dependencies(sample)
    n = len(calls)
    check = check_call - 1 if check_call else None
    kinds = tool_kinds()
    # R2 only when the ask follows a LOOKUP (a read call): the ask depends on how many matches it found. An ask after a
    # write (e.g. turn the TV on, then ask the volume) does not depend on that result, so the plain chain is the plan.
    lookup_ask = {j for j, c in enumerate(calls) if c["name"] == ASK and j > 0 and kinds.get(calls[j - 1]["name"]) == "read"}
    starts = {0} | ({check + 1} if check is not None and check + 1 <= n else set()) | lookup_ask

    results = sample["label"]["tool_results"]
    # what the model is shown: user turns after the prompt's normaliser (numbers said in words become digits, relative
    # dates get a date hint), not the raw transcript; a value visible only in the raw words ("bốn mươi") was wrongly "?"
    now = sample["context"]["current_time"]
    shown = lambda h: build_user_text(h["audio_text"], now) if h.get("role") == "user" and h.get("audio_text") else h.get("text")  # noqa: E731
    user_side = json.dumps([build_user_text(sample["audio_text"], now), sample["context"], [shown(h) for h in sample["history"]]], ensure_ascii=False)

    def render(m: int) -> str:
        seen = user_side + json.dumps([[c["arguments"] for c in calls[:m]], results[:m]], ensure_ascii=False)
        known = lambda v: all(str(x) in seen or json.dumps(x, ensure_ascii=False) in seen for x in scalars(v) if meaningful(x))  # noqa: E731
        out = []
        for k in range(m, n + 1):
            if k == n:
                out.append("end")
                break
            c = calls[k]
            if k in lookup_ask and k != m:  # happy path: the lookup was expected to find exactly one match
                params = c["arguments"].get("params") or []
                out.append(step_text(c["arguments"].get("tool_name") or ASK, {x: None for x in params}, {x: k - 1 for x in params}))
                out.append("end")
                break
            out.append(step_text(c["name"], c["arguments"], deps[k], None if k == m else known))  # step m is called now
            if k == check:
                out.append("replan")
                break
        return "<plan>" + " -> ".join(out) + "</plan>"

    return {m: render(m) for m in sorted(starts)}


def trained_turns(sample: dict, plans: dict[int, str] | None, kinds: dict[str, str]) -> list[dict]:
    """The final-turn assistant/tool messages: one call per assistant message (as the agent loop and the gold replay do).
    plans (experiment A) puts each plan in the content of its message; None (experiment B) writes no plan."""
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    plans = plans or {}
    msgs = []
    for i, (c, r) in enumerate(zip(calls, results)):
        cid = f"gold_{i}"
        msgs.append({"role": "assistant", "content": plans.get(i), "train": True,
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(r, ensure_ascii=False)})
    final = FINAL[outcome(calls, kinds)]
    if len(calls) in plans:
        final = plans[len(calls)] + "\n" + final
    msgs.append({"role": "assistant", "content": final, "train": True})
    return msgs


def load_labels(path: str) -> dict[str, int | None]:
    """{sample id: check_call or None} from the validated Qwen3-8B labels (scripts/plan_branch.py --merge writes
    work/sft/conditional_labels.jsonl). "none" = no conditional anywhere (local tests only)."""
    if path == "none":
        return {}
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r["check_call"] if r.get("valid") and r.get("conditional") else None for r in rows}


def replay_ok(sample: dict, gold_hash: str) -> bool:
    env = tool_env("train", sample["id"])
    res = [env.call(c["name"], c["arguments"]) for c in sample["label"]["tool_calls"]]
    return res == sample["label"]["tool_results"] and env.get_state_hash() == gold_hash


def splits(samples: list[dict]) -> dict[str, str]:
    cv = json.loads((PROJECT / "splits" / "cv5.json").read_text(encoding="utf-8"))
    f1 = sorted(s["id"] for s in samples if cv[s["id"]] == 1)
    val = set(random.Random(0).sample(f1, 60))
    return {s["id"]: "test" if cv[s["id"]] == 0 else "val" if s["id"] in val else "train" for s in samples}


def show(sample: dict, plans: dict[int, str], kinds: dict) -> None:
    print(f"\n== {sample['id']} | {len(sample['label']['tool_calls'])} call | {sample['audio_text']}")
    for name, p in (("A plan + trajectory", plans), ("B trajectory only", None)):
        print(f"   -- {name}: assistant turns trained")
        for m in trained_turns(sample, p, kinds):
            if m["role"] == "tool":
                print(f"         = {m['content'][:160]}")
            elif m.get("tool_calls"):
                f = m["tool_calls"][0]["function"]
                if m["content"]:
                    print(f"      [plan] {m['content']}")
                print(f"      -> {f['name']} {json.loads(f['arguments'])}")
            else:
                print(f"      [final] {m['content']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", help="print these samples' plans and trained turns, write nothing")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ", help="only for the length check (same Qwen3 tokenizer)")
    ap.add_argument("--labels", default=str(OUT / "conditional_labels.jsonl"),
                    help='Qwen3-8B conditional labels (scripts/plan_branch.py --merge); "none" = no conditional (tests only)')
    ap.add_argument("--para", help="kept rewrites (work/synth/paraphrase.kept.jsonl): each adds a train row per dataset, "
                                   'variant "para<n>", with that text as the final user utterance (gold text, no ASR)')
    ap.add_argument("--out", default=str(OUT), help="output dir (default work/sft); use another one with --para")
    a = ap.parse_args()
    labels = load_labels(a.labels)
    para: dict[str, list[tuple[int, str]]] = {}
    if a.para:
        for r in map(json.loads, Path(a.para).read_text(encoding="utf-8").splitlines()):
            para.setdefault(r["id"], []).append((r["n"], r["text"]))
    out_dir = Path(a.out)
    if a.para and out_dir.resolve() == OUT.resolve():
        sys.exit("--para writes a different dataset: pass --out (e.g. work/sft_para), work/sft stays the v2 data")

    samples = load_samples("train")
    tools = {t["name"]: t for x in load_catalog("train")["apps"] for t in x["tools"]}
    kinds = {n: t["kind"] for n, t in tools.items()}
    if a.show is not None:
        by_id = {s["id"]: s for s in samples}
        for sid in a.show:
            s = by_id.get(sid if sid.startswith("train_") else f"train_{int(sid):04d}")
            if s is None:
                print(f"{sid}: not a train id")
            else:
                show(s, plan_segments(s, labels.get(s["id"])), kinds)
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    gold = build_gold("train")
    sp = splits(samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {(d, part): [] for d in ("plan", "traj") for part in ("train", "val", "test")}
    bad_replay, bad_len, n_dep, n_replan = [], [], 0, 0
    for s in samples:
        if not replay_ok(s, gold[s["id"]]["state_hash"]):
            bad_replay.append(s["id"])
            continue
        plan = plan_segments(s, labels.get(s["id"]))
        n_dep += any("=$" in t for t in plan.values())  # counted on the original text
        n_replan += any(t.endswith("-> replan</plan>") for t in plan.values())
        variants = ["gold", "asr"] if sp[s["id"]] == "train" else ["gold"]
        if sp[s["id"]] != "train" and s["id"] in para:
            sys.exit(f"{s['id']}: a rewrite for a {sp[s['id']]} sample (rewrites are train only)")
        variants += [f"para{n}" for n, _ in para.get(s["id"], [])]
        texts = {f"para{n}": t for n, t in para.get(s["id"], [])}
        for v in variants:
            sv = {**s, "audio_text": texts[v]} if v in texts else s  # the rewrite replaces the final user utterance only
            pv = plan_segments(sv, labels.get(s["id"])) if v in texts else plan  # "?" decided on the text the row shows
            p = build_prompt(sv, "train", asr[s["audio"]] if v == "asr" else sv["audio_text"], asr if v == "asr" else None)
            for d, pl in (("plan", pv), ("traj", None)):
                msgs = p["messages"] + trained_turns(s, pl, kinds)
                row = {"id": s["id"], "variant": v, "split": sp[s["id"]], "system": p["system"], "tools": p["tools"], "messages": msgs}
                if d == "plan" and v != "asr":  # longest row of the text: check the length once
                    n = encode(row, tok, 10 ** 6)["n_tokens"]  # lifemate.sft renders exactly what training sees
                    if n > MAX_LEN:
                        bad_len.append((s["id"], n))
                rows[(d, sp[s["id"]])].append(row)
    for (d, part), rs in rows.items():
        (out_dir / f"{d}.{part}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    print(f"replay failures: {len(bad_replay)} | rows over {MAX_LEN} tokens: {len(bad_len)} | plans with a $k: {n_dep} | with a replan: {n_replan} | labels: {a.labels}")
    for (d, part), rs in rows.items():
        print(f"  {d}.{part}: {len(rs)} rows ({len({r['id'] for r in rs})} samples)")
    print(f"-> {out_dir}" + (f" | rewrites: {sum(map(len, para.values()))} from {a.para}" if a.para else ""))


if __name__ == "__main__":
    main()
