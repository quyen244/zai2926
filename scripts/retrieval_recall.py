"""Offline check: could BM25 retrieval shrink the tool list in the LifeMate prompt without dropping a needed tool?

  python scripts/retrieval_recall.py                      # all train samples, every config, table + reports/retrieval_bm25.md
  python scripts/retrieval_recall.py --misses 20          # list more misses of the best app-level config

No model, stdlib only, local CPU. Candidates per sample = exactly the tools the prompt gets today
(lifemate.prompt.build_prompt(...)["tools"], installed apps only, sys.* excluded from ranking: sys.ask_missing_info
stays in the prompt anyway). Gold tools = distinct non-sys names in label.tool_calls.

BM25 (k1=1.5, b=0.75) over one document per candidate tool, IDF computed over that sample's candidates.
  doc "desc"   : tool name split on . and _, app id, app name_vi + description (catalog), tool description
  doc "params" : "desc" + parameter names and parameter descriptions
  tokens       : lowercase + NFC, split on non-word characters; "noacc" variant also strips Vietnamese accents (đ -> d)
  query        : normalize_text(final turn), text = gold (audio_text) or asr (cached PhoWhisper hyp);
                 "+hist" appends the history user turns (same text source)
Metrics
  tool level : sample recall@k = share of samples (with >= 1 gold tool) whose gold tools are ALL in the top-k tools;
               call recall@k = share of gold (sample, tool) pairs in the top-k
  app level  : app score = max (or sum) BM25 of its tools; keep the top-k apps with ALL their tools, plus contacts when
               installed; sample recall and tools kept (mean / median), against today's prompt (all tools)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402

K1, B = 1.5, 0.75
TOOL_K = [3, 5, 8, 10, 15, 20]
APP_K = [1, 2, 3, 4]
REPORT = PROJECT / "reports" / "retrieval_bm25.md"


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokens(s: str, noacc: bool) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    if noacc:
        s = strip_accents(s)
    return [t for t in re.split(r"[\W_]+", s) if t]


def tool_doc(t: dict, apps: dict, with_params: bool) -> str:
    f = t["function"]
    app = f["name"].split(".")[0]
    a = apps.get(app, {})
    parts = [f["name"].replace(".", " ").replace("_", " "), app.replace("_", " "), a.get("name_vi", ""), a.get("description", ""), f.get("description", "")]
    if with_params:
        for k, p in (f.get("parameters", {}).get("properties") or {}).items():
            parts += [k.replace("_", " "), p.get("description", "")]
    return " ".join(parts)


def bm25(query: list[str], docs: list[list[str]]) -> list[float]:
    n = len(docs)
    avg = sum(map(len, docs)) / n
    df = Counter(t for d in docs for t in set(d))
    q = Counter(query)
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * len(d) / avg))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", type=int, default=10)
    a = ap.parse_args()

    samples = load_samples("train")
    apps = {x["id"]: x for x in load_catalog("train")["apps"]}
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    prep = []  # (sample, candidate tool dicts, gold tool names)
    for s in samples:
        tools = [t for t in build_prompt(s, "train", s["audio_text"], None)["tools"] if not t["function"]["name"].startswith("sys.")]
        gold = {c["name"] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}
        prep.append((s, tools, gold))
    n_all = len(prep)
    with_gold = [p for p in prep if p[2]]
    n_tools_today = [len(t) + 1 for _, t, _ in prep]  # + sys.ask_missing_info
    missing_cand = sum(1 for _, t, g in with_gold if not g <= {x["function"]["name"] for x in t})

    def query_text(s: dict, src: str, hist: bool) -> str:
        final = s["audio_text"] if src == "gold" else asr[s["audio"]]
        parts = [final]
        if hist:
            for h in s["history"]:
                if h["role"] == "user":
                    parts.append(h.get("audio_text") if src == "gold" else asr.get(h["audio"], h.get("audio_text", "")) or "")
        return normalize_text(" ".join(p for p in parts if p))

    results = []  # dict per config
    examples = {}
    for src in ("gold", "asr"):
        for hist in (False, True):
            for doc_kind in ("desc", "params"):
                for noacc in (False, True):
                    name = f"{src}{'+hist' if hist else ''} | {doc_kind} | {'noacc' if noacc else 'acc'}"
                    tool_hits = {k: 0 for k in TOOL_K}
                    call_hits = {k: 0 for k in TOOL_K}
                    n_calls = 0
                    app_hits = {(agg, k): 0 for agg in ("max", "sum") for k in APP_K}
                    app_kept = {(agg, k): [] for agg in ("max", "sum") for k in APP_K}
                    misses = {(agg, k): [] for agg in ("max", "sum") for k in APP_K}
                    for s, tools, gold in with_gold:
                        names = [t["function"]["name"] for t in tools]
                        sc = bm25(tokens(query_text(s, src, hist), noacc), [tokens(tool_doc(t, apps, doc_kind == "params"), noacc) for t in tools])
                        order = [names[i] for i in sorted(range(len(names)), key=lambda i: -sc[i])]
                        n_calls += len(gold)
                        for k in TOOL_K:
                            top = set(order[:k])
                            tool_hits[k] += gold <= top
                            call_hits[k] += len(gold & top)
                        by_app: dict[str, list[float]] = {}
                        for nm, x in zip(names, sc):
                            by_app.setdefault(nm.split(".")[0], []).append(x)
                        for agg in ("max", "sum"):
                            ranked = sorted(by_app, key=lambda ap_: -(max if agg == "max" else sum)(by_app[ap_]))
                            for k in APP_K:
                                keep = set(ranked[:k]) | ({"contacts"} & set(by_app))
                                kept = {nm for nm in names if nm.split(".")[0] in keep}
                                ok = gold <= kept
                                app_hits[(agg, k)] += ok
                                app_kept[(agg, k)].append(len(kept) + 1)
                                if not ok:
                                    misses[(agg, k)].append((s["id"], query_text(s, src, False), sorted(gold - kept)))
                    n = len(with_gold)
                    r = {"name": name, "tool": {k: tool_hits[k] / n for k in TOOL_K}, "call": {k: call_hits[k] / n_calls for k in TOOL_K},
                         "app": {f"{agg}@{k}": (app_hits[(agg, k)] / n, st.mean(app_kept[(agg, k)]), st.median(app_kept[(agg, k)])) for agg in ("max", "sum") for k in APP_K}}
                    results.append(r)
                    examples[name] = misses

    n = len(with_gold)
    lines = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# BM25 tool retrieval: offline recall on train")
    out()
    out(f"Train samples: {n_all}; with at least one non-sys gold tool: {n}; without (gold empty or only sys.ask_missing_info): {n_all - n}.")
    out(f"Samples whose gold tool is not among today's candidates at all: {missing_cand}.")
    out(f"Before (all tools, today's prompt): sample recall 100% by construction, tools per prompt mean {st.mean(n_tools_today):.1f}, median {st.median(n_tools_today):.0f} (incl. sys.ask_missing_info).")
    out()
    out("## Tool level: sample recall@k (all gold tools in top-k)")
    out()
    out("| config | " + " | ".join(f"@{k}" for k in TOOL_K) + " | call recall@10 |")
    out("|---|" + "---|" * (len(TOOL_K) + 1))
    for r in results:
        out(f"| {r['name']} | " + " | ".join(f"{100 * r['tool'][k]:.1f}" for k in TOOL_K) + f" | {100 * r['call'][10]:.1f} |")
    out()
    out("## App level: keep top-k apps (all their tools) + contacts; sample recall % / mean tools kept / median tools kept")
    out()
    keys = [f"{agg}@{k}" for agg in ("max", "sum") for k in APP_K]
    out("| config | " + " | ".join(keys) + " |")
    out("|---|" + "---|" * len(keys))
    for r in results:
        out(f"| {r['name']} | " + " | ".join(f"{100 * r['app'][x][0]:.1f} / {r['app'][x][1]:.1f} / {r['app'][x][2]:.0f}" for x in keys) + " |")
    out()

    # best app-level config on ASR text (what the pipeline will see): smallest k reaching >= 99%, then fewest tools kept
    asr_results = [r for r in results if r["name"].startswith("asr")]
    cands = [(r, x) for r in asr_results for x in keys if r["app"][x][0] >= 0.99]
    out("## Verdict")
    out()
    if cands:
        r, x = min(cands, key=lambda c: (c[0]["app"][c[1]][1], -c[0]["app"][c[1]][0]))
    else:
        r, x = max(((r, x) for r in asr_results for x in keys), key=lambda c: (c[0]["app"][c[1]][0], -c[0]["app"][c[1]][1]))
    rec, mean_k, med_k = r["app"][x]
    best_tool = max(((rr, k) for rr in asr_results for k in TOOL_K), key=lambda c: (c[0]["tool"][c[1]], -c[1]))
    gold_side = [rr for rr in results if rr["name"] == r["name"].replace("asr", "gold", 1)][0]
    out(f"Best app-level config on ASR text: {r['name']}, {x}: sample recall {100 * rec:.1f}%, tools kept mean {mean_k:.1f}, median {med_k:.0f} (today median {st.median(n_tools_today):.0f}).")
    out(f"Same config on the reference transcript: sample recall {100 * gold_side['app'][x][0]:.1f}%.")
    out(f"{'It reaches' if rec >= 0.99 else 'No app-level config reaches'} the 99% sample-recall bar{' ' if rec >= 0.99 else ' on ASR text; the best one is shown'}.")
    out(f"Verdict: BM25 app-level retrieval is {'safe at this setting' if rec >= 0.99 else 'NOT safe'}: "
        f"{'' if rec >= 0.99 else f'it drops a needed tool in {100 * (1 - rec):.1f}% of samples, and each such sample scores 0'}.")
    out(f"Tool-level top-k is weaker: best on ASR text is {best_tool[0]['name']} @{best_tool[1]} with {100 * best_tool[0]['tool'][best_tool[1]]:.1f}% sample recall; top-3 and top-5 miss far more.")
    miss = examples[r["name"]][(x.split("@")[0], int(x.split("@")[1]))]
    missed = Counter(t for _, _, m in miss for t in m)
    out(f"Misses of the best config: {len(miss)} of {n} samples. Most often missing gold tools: "
        + ", ".join(f"`{t}` {c}" for t, c in missed.most_common(8)) + ".")
    out("Per-sample misses (ids and queries) are printed to the screen only: the repo is public.")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n(screen only) first {min(a.misses, len(miss))} misses of the best config:")
    for sid, q, m in miss[: a.misses]:
        print(f"  {sid} | {q} | missing {', '.join(m)}")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
