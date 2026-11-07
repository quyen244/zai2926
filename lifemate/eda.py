"""Error classification of a cached run, shared by scripts/error_eda.py, error_ids.py and compare_sample.py.

A sample can belong to several groups:
  1 wrong_args     a tool called by its right name, but no call of it matches the gold exactly
    1a extra_args  the closest call has an argument the gold does not have
    1b missing_args  the closest call lacks an argument the gold has
    1c wrong_value the closest call has a gold argument with a different value (fuzzy params by token F1)
  2 under          a gold tool name is missing from the prediction
  3 over           the prediction has a tool name the gold does not have
    3a extra_ask   sys.ask_missing_info that the gold does not have
  4 wrong_tool     an unrequested tool (not sys) is called while a gold tool is missing
  5 no_tool_ok / no_tool_wrong   gold is []: correct when the model also made no call
    5b no_call     gold needs calls and the model made none
  6 lookup_skip    gold uses a lookup tool and the prediction does not
  7 multi_step     gold has 3+ calls or the request is conditional ("nếu")
  hash             final state differs from gold
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lifemate.data import PROJECT, fuzzy_params, load_samples
from lifemate.gold import build_gold
from lifemate.scoring import FUZZY_THRESHOLD, _args, calls_match, score_sample, token_f1

CACHE_DIR = PROJECT / "work" / "kaggle" / "cache"
LOOKUPS = ("contacts.search", "calendar.list_events", "banking.list_beneficiaries")
GROUPS = [("wrong_args", "1. Đúng tool, sai tham số"), ("extra_args", "1a. có tham số thừa"), ("missing_args", "1b. thiếu tham số"),
          ("wrong_value", "1c. có tham số sai giá trị"), ("under", "2. Gọi thiếu"),
          ("over", "3. Gọi thừa"), ("extra_ask", "3a. trong đó hỏi lại thừa"), ("wrong_tool", "4. Chọn nhầm tool"),
          ("no_tool_wrong", "5. Gold rỗng nhưng vẫn gọi"), ("no_tool_ok", "5. Gold rỗng, không gọi (đúng)"),
          ("no_call", "5b. Cần gọi nhưng không gọi gì"), ("lookup_skip", "6. Bỏ tool tra cứu"), ("multi_step", "7. Nhiều bước / có điều kiện"),
          ("hash", "Lệch state hash")]
TITLE = dict(GROUPS)
REASON = {"wrong_args": "đúng tool, sai tham số", "extra_args": "thừa tham số", "missing_args": "thiếu tham số", "wrong_value": "sai giá trị tham số",
          "under": "gọi thiếu tool", "over": "gọi thừa tool", "extra_ask": "hỏi lại thừa", "wrong_tool": "chọn nhầm tool",
          "no_tool_wrong": "gold rỗng nhưng vẫn gọi", "no_call": "cần gọi nhưng không gọi gì", "lookup_skip": "bỏ tool tra cứu",
          "hash": "lệch state hash"}


def arg_diff(p: dict, g: dict, fz: dict) -> dict:
    """Per-argument difference of two calls of the same tool: extra / missing / wrong-value keys."""
    f, pa, ga = fz.get(g["name"], set()), _args(p), _args(g)
    wrong = [k for k in ga if k in pa and not (token_f1(pa[k], ga[k]) > FUZZY_THRESHOLD if k in f else pa[k] == ga[k])]
    return {"extra": sorted(set(pa) - set(ga)), "missing": sorted(set(ga) - set(pa)), "wrong": sorted(wrong)}


def arg_pairs(pred: list[dict], gold: list[dict], fz: dict) -> list[tuple[dict, dict, dict]]:
    """For each tool name in both lists with no exact match: the closest (pred, gold) pair and its diff."""
    out = []
    for n in sorted({c["name"] for c in pred} & {c["name"] for c in gold}):
        ps, gs = [p for p in pred if p["name"] == n], [g for g in gold if g["name"] == n]
        if any(calls_match(p, g, fz) for p in ps for g in gs):
            continue
        best = min(((p, g, arg_diff(p, g, fz)) for p in ps for g in gs), key=lambda t: sum(len(v) for v in t[2].values()))
        out.append(best)
    return out


def classify(pred: list[dict], gold: list[dict], text: str, hash_ok: bool, fz: dict) -> tuple[list[str], dict]:
    pn, gn = Counter(c["name"] for c in pred), Counter(c["name"] for c in gold)
    out, extra = [], {}
    pairs = arg_pairs(pred, gold, fz)
    if pairs:
        out.append("wrong_args")
        for key, lab in (("extra", "extra_args"), ("missing", "missing_args"), ("wrong", "wrong_value")):
            if any(d[key] for _, _, d in pairs):
                out.append(lab)
    missing, added = set(gn - pn), set(pn - gn)
    if missing:
        out.append("under")
    if added:
        out.append("over")
    if "sys.ask_missing_info" in added:
        out.append("extra_ask")
    wrong = sorted(n for n in added if not n.startswith("sys."))
    if wrong and missing:
        out.append("wrong_tool")
        extra["confusions"] = [f"{w} thay {m}" for w in wrong for m in sorted(missing) if w.split(".")[0] == m.split(".")[0]]
    if not gold:
        out.append("no_tool_ok" if not pred else "no_tool_wrong")
    elif not pred:
        out.append("no_call")
    skipped = [t for t in LOOKUPS if t in gn and t not in pn]
    if skipped:
        out.append("lookup_skip")
        extra["skipped"] = skipped
    if len(gold) >= 3 or "nếu" in text.lower():
        out.append("multi_step")
    if not hash_ok:
        out.append("hash")
    return out, extra


def variant_of(f: Path) -> str:
    v = "B0 (gold)" if ".gold." in f.name else "B1 (asr)" if ".asr." in f.name else f.stem
    return f"{f.name.split('.')[0]} {v}"


def load_runs(cache_dir: Path = CACHE_DIR) -> dict[tuple[str, Path], list[dict]]:
    """Every cache file scored and classified: {(variant, file): [per-sample row]}, B0 before B1."""
    samples = {s["id"]: s for s in load_samples("train")}
    gold_hash, fz = build_gold("train"), fuzzy_params("train")
    files = sorted(cache_dir.glob("*.jsonl"), key=lambda f: (".asr." in f.name, f.name))
    if not files:
        raise SystemExit(f"no cache file in {cache_dir}")
    asr_file = PROJECT / "work" / "asr" / "phowhisper" / "train.jsonl"
    asr = {}
    if asr_file.is_file():
        asr = {(j := json.loads(l))["audio"]: j["hyp"] for l in asr_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    runs = {}
    for f in files:
        res = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = samples[r["id"]]
            g = s["label"]["tool_calls"]
            sc = score_sample(r["tool_calls"], r["state_hash"], g, gold_hash[r["id"]]["state_hash"], fz)
            labs, extra = classify(r["tool_calls"], g, s["audio_text"], sc.hash_ok, fz)
            res.append({"id": r["id"], "score": sc.score, "labels": labs, "pred": r["tool_calls"], "gold": g, "text": s["audio_text"],
                        "asr": asr.get(s["audio"]), "pairs": arg_pairs(r["tool_calls"], g, fz), **extra})
        runs[(variant_of(f), f)] = res
    return runs
