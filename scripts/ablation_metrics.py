"""Metrics of the plan ablation from the eval caches: plan + trajectory SFT vs trajectory-only SFT vs the untuned base.

  python scripts/ablation_metrics.py                  # -> reports/plan_sft_ablation.md (+ reports/plan_sft_loss.svg), aggregates only
  python scripts/ablation_metrics.py --examples 3     # also write representative trajectories to work/eda/plan_sft_examples.md

Inputs: work/kaggle/cache/Qwen3-4B{,-planSFT,-trajSFT}.{gold,fail}.sc6.<hash>.jsonl (kaggle/plan_sft_eval/, merged with
scripts/kaggle_merge_cache.py) and the training outputs work/kaggle/plan_sft_train/out/{plan,traj}/{log.jsonl,stats.json}.
Definitions (per system; "gold" variant = 309 test samples, reference text):
  official score       lifemate.scoring (hash gate x call F1), x100; paired bootstrap 95% CI of the difference
  task success         sample score == 1 (state hash and every call right)
  tool selection       micro F1 of tool names (multisets per sample, summed)
  argument exact match matched calls / same-name call pairs (official argument rules)
  sequence accuracy    predicted tool-name sequence == gold sequence (gold not empty)
  trajectory success   sequence right AND each call matches its gold call position by position
  tool efficiency      predicted calls / gold calls (gold not empty); mean calls per sample
  no-tool accuracy     gold empty and no call predicted
  error recovery       natural failures only (no injection): runs where a call returned ok false / done false, share
                       that called the failed tool again later, and their score vs runs without a failure
  plan metrics (A)     on every GENERATED plan "<plan>a(x=..) -> b(y=$1) -> end|replan</plan>" against the plans the gold
                       trajectory gives (scripts/plan_data.py R1-R4): validity, first-plan correctness, plan-to-execution
                       consistency, dependency accuracy ($k reference F1, TinyAgent-style), replan at the right check
                       (conditional samples), unnecessary planning
  operational          input tokens (every LLM request, Qwen3 template), output tokens (every reply), plan tokens,
                       wall seconds per sample on the T4 (relative only; the scoring A30 is faster)
Categories: single-step, multi-step, conditional, data-dependent, intent correction (self_repair), tool failure (natural
failures, section 6), missing info (gold asks), no-tool (gold empty), unnecessary-call avoidance (gold has no write: share of samples
with a predicted write).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, fuzzy_params, load_catalog, load_samples  # noqa: E402
from lifemate.eda import CACHE_DIR, GROUPS, TITLE, classify  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from lifemate.scoring import calls_match, max_matching, score_sample  # noqa: E402
from plan_data import OUT as SFT_DIR, load_labels, make_plan, plan_segments  # noqa: E402

SYSTEMS = [("Qwen3-4B-planSFT", "A: Plan + Trajectory SFT"), ("Qwen3-4B-trajSFT", "B: Trajectory-only SFT"), ("Qwen3-4B", "Base (no SFT)")]
TRAIN_OUT = PROJECT / "work" / "kaggle" / "plan_sft_train" / "out"
REPORT = PROJECT / "reports" / "plan_sft_ablation.md"
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S)


def latest(label: str, variant: str, cache_dir: Path = CACHE_DIR) -> Path | None:
    fs = sorted(cache_dir.glob(f"{label}.{variant}.sc*.jsonl"), key=lambda f: f.stat().st_mtime)
    return fs[-1] if fs else None


def rows_of(f: Path | None) -> dict[str, dict]:
    if f is None:
        return {}
    return {(r := json.loads(l))["id"]: r for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}


def parse_one(text: str) -> dict:
    """One plan "<plan>a(x=1) -> b(y=$1) -> end|replan</plan>" -> {"steps": [(tool, {param: k})], "ends": "end"|"replan"|None}."""
    parts = [x.strip() for x in text.split(" -> ")]
    tail = parts[-1] if parts and parts[-1] in ("end", "replan") else None
    steps = [(x.split("(", 1)[0].strip(), {pp: int(k) for pp, k in re.findall(r"(\w+)=\$(\d+)", x)}) for x in (parts[:-1] if tail else parts) if x]
    return {"steps": steps, "ends": tail}


def parse_plans(replies: list[dict]) -> list[tuple[int, dict, str]]:
    """Every plan the model wrote in this run: (reply index, parsed plan, raw text)."""
    out = []
    for i, rep in enumerate(replies or []):
        for m in PLAN_RE.finditer(rep.get("content") or ""):
            out.append((i, parse_one(m.group(1)), m.group(0)))
    return out


def bootstrap(d: list[float], n: int = 10000) -> tuple[float, float]:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]


def loss_svg(logs: dict[str, list[dict]], path: Path) -> None:
    """Training loss (and val loss points) per optimizer step, one line per experiment, inline SVG (no plotting lib)."""
    W, H, P = 720, 300, 46
    pts = {k: [(r["step"], r["loss"]) for r in v if "loss" in r] for k, v in logs.items()}
    val = {k: [(r["step"], r["eval_loss"]) for r in v if "eval_loss" in r] for k, v in logs.items()}
    xs = [s for v in pts.values() for s, _ in v] or [1]
    ys = [y for v in list(pts.values()) + list(val.values()) for _, y in v] or [0, 1]
    x0, x1, y0, y1 = 0, max(xs), min(ys), max(ys)
    sx = lambda x: P + (W - 2 * P) * (x - x0) / max(1e-9, x1 - x0)  # noqa: E731
    sy = lambda y: H - P + (2 * P - H) * (y - y0) / max(1e-9, y1 - y0)  # noqa: E731
    color = {"plan": "#2f6fd6", "traj": "#c9423a"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>', f'<line x1="{P}" y1="{H - P}" x2="{W - P}" y2="{H - P}" stroke="#333"/>',
           f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H - P}" stroke="#333"/>',
           f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle">optimizer step</text>',
           f'<text x="12" y="{P - 14}">loss ({y0:.2f} to {y1:.2f})</text>']
    for k, v in pts.items():
        if v:
            out.append(f'<polyline fill="none" stroke="{color.get(k, "#333")}" stroke-width="1.6" points="' + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in v) + '"/>')
        for x, y in val.get(k, []):
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color.get(k, "#333")}"/>')
    for i, k in enumerate(pts):
        out.append(f'<text x="{W - P - 200}" y="{P + 16 * i}" fill="{color.get(k, "#333")}">{"A plan + trajectory" if k == "plan" else "B trajectory only"} (dots: val)</text>')
    path.write_text("\n".join(out + ["</svg>"]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=0, help="write N success and N failure trajectories per system to work/eda/")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ", help="same Qwen3 tokenizer, for token counts")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--report", type=Path, default=REPORT, help="output path (tests write elsewhere)")
    ap.add_argument("--labels", type=Path, default=SFT_DIR / "conditional_labels.jsonl", help="Qwen3-8B conditional labels")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    samples = {s["id"]: s for s in load_samples("train")}
    gold_hash, fz = build_gold("train"), fuzzy_params("train")
    tools_cat = {t["name"]: t for x in load_catalog("train")["apps"] for t in x["tools"]}
    kinds = {n: t["kind"] for n, t in tools_cat.items()}
    tags = {(r := json.loads(l))["id"]: set(r["linguistic"]) | set(r["structural"]) for l in (PROJECT / "work/tags.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    runs = {(lab, "gold"): rows_of(latest(lab, "gold", a.cache_dir)) for lab, _ in SYSTEMS}
    ids = sorted(set.intersection(*[set(runs[(lab, "gold")]) for lab, _ in SYSTEMS if runs[(lab, "gold")]] or [set()]))
    if not ids:
        raise SystemExit("no eval cache for the ablation systems in work/kaggle/cache/")

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    per = {}  # (label) -> per-sample metrics
    for lab, _ in SYSTEMS:
        res = runs[(lab, "gold")]
        if not res:
            continue
        rows = []
        for i in ids:
            r, s = res[i], samples[i]
            g = s["label"]["tool_calls"]
            p = r["tool_calls"]
            sc = score_sample(p, r["state_hash"], g, gold_hash[i]["state_hash"], fz)
            pn, gn = Counter(c["name"] for c in p), Counter(c["name"] for c in g)
            same = sum((pn & gn).values())
            seq_ok = [c["name"] for c in p] == [c["name"] for c in g]
            prompt = build_prompt(s, "train", s["audio_text"], None)
            msgs = [{"role": "system", "content": prompt["system"]}] + prompt["messages"]
            in_tok = out_tok = 0
            for rep in r.get("replies", []):  # every request = the prompt so far; every reply = generated text
                text = tok.apply_chat_template(msgs, tools=prompt["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                in_tok += ntok(text)
                gen = (rep.get("content") or "") + "".join(
                    f'<tool_call>\n{{"name": "{c["function"]["name"]}", "arguments": {c["function"]["arguments"]}}}\n</tool_call>' for c in rep.get("tool_calls") or [])
                out_tok += ntok(gen)
                msgs.append({"role": "assistant", "content": rep.get("content") or "", "tool_calls": [
                    {**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"] or "{}")}} for c in rep.get("tool_calls") or []]} if rep.get("tool_calls") else
                    {"role": "assistant", "content": rep.get("content") or ""})
                for c in rep.get("tool_calls") or []:
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": "{}"})  # result size ~ small vs prompt; counted as a placeholder
            plans = parse_plans(r.get("replies"))
            row = {"id": i, "score": sc.score, "hash": sc.hash_ok, "success": sc.score >= 0.999, "pn": pn, "gn": gn, "same": same,
                   "matched": max_matching(p, g, fz), "seq_ok": seq_ok and bool(g),
                   "traj_ok": seq_ok and bool(g) and all(calls_match(x, y, fz) for x, y in zip(p, g)), "n_pred": len(p), "n_gold": len(g),
                   "in_tok": in_tok, "out_tok": out_tok, "sec": r.get("sec", 0.0), "plans": plans,
                   "plan_tok": sum(ntok(t) for _, _, t in plans),
                   "labels": classify(p, g, s["audio_text"], sc.hash_ok, fz)[0], "pred_write": any(kinds.get(c["name"]) == "write" for c in p),
                   "tools_offered": {t["function"]["name"] for t in prompt["tools"]}}
            rows.append(row)
        per[lab] = rows

    cats = {
        "1. single-step": lambda s: len([c for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")]) == 1,
        "2. sequential multi-step": lambda s: len(s["label"]["tool_calls"]) >= 2,
        "3. conditional": lambda s: "conditional" in tags.get(s["id"], set()),
        "4. data-dependent arguments": lambda s: any(st_["depends_on"] for st_ in make_plan(s, tools_cat)["steps"]),
        "5. user intent correction": lambda s: "self_repair" in tags.get(s["id"], set()),
        "7. missing information": lambda s: any(c["name"] == "sys.ask_missing_info" for c in s["label"]["tool_calls"]),
        "8. no-tool decision": lambda s: not s["label"]["tool_calls"],
        "9. unnecessary-call avoidance (gold has no write)": lambda s: not any(kinds.get(c["name"]) == "write" for c in s["label"]["tool_calls"]),
    }
    lines = []

    def out(x: str = "") -> None:
        print(x)
        lines.append(x)

    present = [(lab, name) for lab, name in SYSTEMS if lab in per]
    hdr = "| Metric | " + " | ".join(n for _, n in present) + " |"
    sep = "|---|" + "---:|" * len(present)
    pct = lambda xs: f"{100 * sum(xs) / max(1, len(xs)):.1f}%"  # noqa: E731
    out("# Plan + Trajectory SFT vs Trajectory-only SFT (ablation)")
    out()
    out(f"Generated by `scripts/ablation_metrics.py` from the eval caches; {len(ids)} held-out test samples (cv5 fold 0), reference transcript, step cap 6.")
    out("Base model Qwen/Qwen3-4B (FP16 at inference, 4-bit NF4 during QLoRA training); A and B differ only in the plan text of the first assistant turn.")
    out("Aggregates only; per-sample trajectories are in `work/eda/plan_sft_examples.md` (git-ignored).")
    out()
    out("## Required comparison")
    out()
    out(hdr)
    out(sep)

    def row_(name: str, f) -> None:
        out(f"| {name} | " + " | ".join(f(per[lab]) for lab, _ in present) + " |")

    row_("Official score (x100)", lambda R: f"{100 * st.mean(r['score'] for r in R):.2f}")
    row_("Task Success Rate", lambda R: pct([r["success"] for r in R]))
    row_("Tool Selection Accuracy (name F1)", lambda R: f"{100 * 2 * sum(r['same'] for r in R) / max(1, sum(r['n_pred'] + r['n_gold'] for r in R)):.1f}%")
    row_("Argument Exact Match", lambda R: f"{100 * sum(r['matched'] for r in R) / max(1, sum(r['same'] for r in R)):.1f}%")
    row_("Sequence Accuracy", lambda R: pct([r["seq_ok"] for r in R if r["n_gold"]]))
    row_("Trajectory Success Rate", lambda R: pct([r["traj_ok"] for r in R if r["n_gold"]]))
    row_("Error Recovery Rate (natural failures, see section 6)", lambda R: "see below")
    row_("No-tool Accuracy", lambda R: pct([r["n_pred"] == 0 for r in R if r["n_gold"] == 0]))
    row_("Average Tool Calls (gold mean %.2f)" % st.mean(r["n_gold"] for r in next(iter(per.values()))), lambda R: f"{st.mean(r['n_pred'] for r in R):.2f}")
    row_("Tool Efficiency (pred / gold calls)", lambda R: f"{st.mean(r['n_pred'] / r['n_gold'] for r in R if r['n_gold']):.2f}")
    row_("State hash match", lambda R: pct([r["hash"] for r in R]))
    row_("Average Input Tokens (all requests)", lambda R: f"{st.mean(r['in_tok'] for r in R):.0f}")
    row_("Average Output Tokens", lambda R: f"{st.mean(r['out_tok'] for r in R):.0f}")
    row_("Average Planning Tokens", lambda R: f"{st.mean(r['plan_tok'] for r in R):.0f}")
    row_("End-to-end Latency (T4 s/sample, relative)", lambda R: f"{st.mean(r['sec'] for r in R):.1f}")
    out()
    out("## Paired differences (official score x100, 95% bootstrap CI)")
    out()
    by = {lab: {r["id"]: r["score"] for r in R} for lab, R in per.items()}
    for x, y in (("Qwen3-4B-planSFT", "Qwen3-4B-trajSFT"), ("Qwen3-4B-planSFT", "Qwen3-4B"), ("Qwen3-4B-trajSFT", "Qwen3-4B")):
        if x in by and y in by:
            d = [100 * (by[x][i] - by[y][i]) for i in ids]
            lo, hi = bootstrap(d)
            out(f"- {dict(SYSTEMS)[x]} minus {dict(SYSTEMS)[y]}: {st.mean(d):+.2f} (CI {lo:+.2f} to {hi:+.2f}); better on {sum(v > 0 for v in d)}, worse on {sum(v < 0 for v in d)} samples.")
    out()
    out("## Error EDA (share of the test samples per group, lifemate/eda.py)")
    out()
    out(hdr.replace("Metric", "Group"))
    out(sep)
    for key, _ in GROUPS:
        out(f"| {TITLE[key]} | " + " | ".join(pct([key in r["labels"] for r in per[lab]]) for lab, _ in present) + " |")
    out()
    out("## Per category (official score x100 / task success / n)")
    out()
    out(hdr.replace("Metric", "Category"))
    out(sep)
    for cname, f in cats.items():
        sel = {i for i in ids if f(samples[i])}
        if not sel:
            continue
        cells = []
        for lab, _ in present:
            R = [r for r in per[lab] if r["id"] in sel]
            extra = f", writes {pct([r['pred_write'] for r in R])}" if cname.startswith("9") else ""
            cells.append(f"{100 * st.mean(r['score'] for r in R):.1f} / {pct([r['success'] for r in R])} / {len(R)}{extra}")
        out(f"| {cname} | " + " | ".join(cells) + " |")
    # 6. tool failure and recovery, on natural failures (the run is replayed from its cached replies)
    out()
    out("## 6. Tool failure and recovery (natural failures: a call returned ok false or done false)")
    out()
    out("| System | runs with a failed call | of which the failed tool was called again | score of runs with a failure | score of runs without |")
    out("|---|---:|---:|---:|---:|")
    from lifemate.agent import run_sample
    from trace_sample import CachedReplies
    for lab, name in present:
        with_f, retry, sc_f, sc_ok = 0, 0, [], []
        for r in per[lab]:
            raw, s = runs[(lab, "gold")][r["id"]], samples[r["id"]]
            ev = []
            run_sample(s, "train", s["audio_text"], CachedReplies(raw["replies"]), trace=lambda e, d: e == "call" and ev.append((d["name"], d["result"])))
            bad = [k for k, (n, res) in enumerate(ev) if res.get("ok") is False or res.get("done") is False]
            if bad:
                with_f += 1
                retry += any(ev[j][0] == ev[bad[0]][0] for j in range(bad[0] + 1, len(ev)))
                sc_f.append(r["score"])
            else:
                sc_ok.append(r["score"])
        out(f"| {name} | {with_f} ({100 * with_f / len(per[lab]):.1f}%) | {retry} ({100 * retry / max(1, with_f):.1f}%) | "
            f"{100 * st.mean(sc_f) if sc_f else float('nan'):.1f} | {100 * st.mean(sc_ok) if sc_ok else float('nan'):.1f} |")
    # plan metrics (experiment A): every plan the model wrote, against the plans the gold trajectory gives (plan_data R1-R4)
    if "Qwen3-4B-planSFT" in per:
        R = per["Qwen3-4B-planSFT"]
        cond = load_labels(str(a.labels)) if a.labels.is_file() else {}
        valid, first_ok, consist, unnec, replan_ok, n_cond = [], [], [], [], [], 0
        ref_tp = ref_fp = ref_fn = 0
        for r in R:
            s_ = samples[r["id"]]
            plans = r["plans"]
            executed = [c["name"] for c in runs[("Qwen3-4B-planSFT", "gold")][r["id"]]["tool_calls"]]
            ok = bool(plans) and all(p["ends"] and all(t in r["tools_offered"] for t, _ in p["steps"]) for _, p, _ in plans)
            valid.append(ok)
            gold_plans = {m: parse_one(t[len("<plan>"):-len("</plan>")]) for m, t in plan_segments(s_, cond.get(r["id"])).items()}
            g0 = gold_plans.get(0) or next(iter(gold_plans.values()))
            p0 = plans[0][1] if plans else {"steps": [], "ends": None}
            first_ok.append(ok and [t for t, _ in p0["steps"]] == [t for t, _ in g0["steps"]] and p0["ends"] == g0["ends"])
            # the executed calls follow the plans: each plan's steps are a prefix of what was executed from that point
            seq = [t for _, p, _ in plans for t, _ in p["steps"]]
            consist.append(ok and executed[: len(seq)] == seq[: len(executed)])
            if not s_["label"]["tool_calls"]:
                unnec.append(any(p["steps"] for _, p, _ in plans))
            if cond.get(r["id"]):
                n_cond += 1
                replan_ok.append(ok and p0["ends"] == "replan" and [t for t, _ in p0["steps"]] == [t for t, _ in g0["steps"]])
            if ok and [t for t, _ in p0["steps"]] == [t for t, _ in g0["steps"]]:  # $k references, on aligned first plans
                pr = {(k, pp, v) for k, (_, d) in enumerate(p0["steps"]) for pp, v in d.items()}
                gr = {(k, pp, v) for k, (_, d) in enumerate(g0["steps"]) for pp, v in d.items()}
                ref_tp += len(pr & gr)
                ref_fp += len(pr - gr)
                ref_fn += len(gr - pr)
        out()
        out("## Plan metrics (A only, measured on the generated plans)")
        out()
        out(f"- Plan validity (every plan ends with end/replan, every tool offered): {pct(valid)}")
        out(f"- First plan correctness (tool chain and ending equal the gold-derived first plan): {pct(first_ok)}")
        out(f"- Plan-to-execution consistency (executed calls follow the written plans): {pct(consist)}")
        out(f"- Dependency accuracy ($k references, F1, on samples whose first plan has the gold tool chain): {100 * 2 * ref_tp / max(1, 2 * ref_tp + ref_fp + ref_fn):.1f}%")
        out(f"- Replan at the right check (conditional test samples, Qwen3-8B labels): {pct(replan_ok)} of {n_cond}")
        out(f"- Unnecessary planning (gold calls nothing, a plan names a tool): {pct(unnec)} of {len(unnec)}")
    # training
    logs = {d: [json.loads(l) for l in (TRAIN_OUT / d / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            for d in ("plan", "traj") if (TRAIN_OUT / d / "log.jsonl").is_file()}
    stats = {d: json.loads((TRAIN_OUT / d / "stats.json").read_text(encoding="utf-8")) for d in ("plan", "traj") if (TRAIN_OUT / d / "stats.json").is_file()}
    if stats:
        out()
        out("## Training (QLoRA, identical settings)")
        out()
        out("| | A plan + trajectory | B trajectory only |")
        out("|---|---:|---:|")
        for k, title in (("rows", "train rows"), ("steps", "optimizer steps"), ("tokens_seen", "tokens seen"), ("supervised_tokens_seen", "supervised tokens seen"),
                         ("supervised_per_row", "supervised tokens per row"), ("train_wall_s", "train wall time (s)"), ("gpu_hours", "GPU hours (one T4)"),
                         ("peak_mem_gb", "peak GPU memory (GB)"), ("tokens_per_s", "tokens per second"), ("final_val_loss", "final val loss")):
            cell = lambda d: (f"{stats[d][k]:.3f}" if isinstance(stats[d].get(k), float) else str(stats[d].get(k))) if d in stats else "-"  # noqa: E731
            out(f"| {title} | {cell('plan')} | {cell('traj')} |")
        if logs:
            loss_svg(logs, a.report.with_name("plan_sft_loss.svg"))
            out()
            out("Loss curves: `reports/plan_sft_loss.svg` (lines: training loss per step; dots: validation loss). The two losses are not directly comparable: A also predicts the plan tokens.")
    a.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {a.report}")
    if a.examples:
        ex = ["# Plan ablation: representative trajectories (per-sample data, git-ignored)", ""]
        for lab, name in present:
            R = sorted(per[lab], key=lambda r: r["id"])
            for kind, sel in (("success", [r for r in R if r["success"] and r["n_gold"] >= 2]), ("failure", [r for r in R if not r["success"] and r["n_gold"] >= 2])):
                for r in sel[: a.examples]:
                    ex.append(f"## {name} | {kind} | {r['id']} | score {r['score']:.2f}")
                    ex.append(f"    {samples[r['id']]['audio_text']}")
                    for _, _, t in r["plans"]:
                        ex.append(f"    [plan] {t}")
                    for c in runs[(lab, "gold")][r["id"]]["tool_calls"]:
                        ex.append(f"    -> {c['name']} {c['arguments']}")
                    ex.append("    gold:")
                    for c in samples[r["id"]]["label"]["tool_calls"]:
                        ex.append(f"    -> {c['name']} {c['arguments']}")
                    ex.append("")
        dst = PROJECT / "work" / "eda" / "plan_sft_examples.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(ex) + "\n", encoding="utf-8")
        print(f"-> {dst}")


if __name__ == "__main__":
    main()
