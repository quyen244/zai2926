"""Prompt size per error group of a cached Kaggle run: are the "right tools, wrong arguments" samples the long prompts?

  python scripts/prompt_by_error.py                       # Qwen3-4B-AWQ, B0 (gold) and B1 (asr)
  python scripts/prompt_by_error.py --model QVikhr --variant gold

Groups (per sample of the run):
  seq_ok_extra_missing  predicted tool names == gold tool names in the same order, and at least one call has an
                        extra or a missing argument (closest same-tool pair, lifemate.eda.arg_pairs)
  seq_ok_value_only     same tool sequence, no extra/missing argument, but some argument value is wrong
  seq_ok_exact          same tool sequence and every call matches gold
  seq_wrong             tool sequence differs from gold (under, over, wrong tool, no call, ...)
Samples whose gold has no tool call are left out (no sequence to get right).
Prompt = the first LLM request, built as the agent builds it (lifemate.prompt.build_prompt; B0 reference text, B1 cached
ASR for the final and history turns). Size = Qwen3 tokens after the chat template (tokenizer on disk), chars, tools
(incl. sys.ask_missing_info), apps (incl. sys), history turns. Aggregates only, screen only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.eda import CACHE_DIR, load_runs  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402

GROUPS = [("seq_ok_extra_missing", "đúng chuỗi tool, thừa/thiếu tham số"), ("seq_ok_value_only", "đúng chuỗi tool, chỉ sai giá trị"),
          ("seq_ok_exact", "đúng chuỗi tool, đúng hết"), ("seq_wrong", "sai chuỗi tool"), ("all", "tất cả (gold có tool)")]


def group_of(r: dict) -> str:
    if [c["name"] for c in r["pred"]] != [c["name"] for c in r["gold"]]:
        return "seq_wrong"
    diffs = [d for _, _, d in r["pairs"]]
    if any(d["extra"] or d["missing"] for d in diffs):
        return "seq_ok_extra_missing"
    return "seq_ok_value_only" if diffs else "seq_ok_exact"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-AWQ", help="substring of the cache file name")
    ap.add_argument("--variant", choices=["gold", "asr", "all"], default="all")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    a = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        tok = None
        print(f"(no tokenizer on disk, chars only: {type(e).__name__})")
    samples = {s["id"]: s for s in load_samples("train")}
    asr = {(j := json.loads(l))["audio"]: j["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    for (v, f), res in load_runs(a.cache_dir).items():
        if a.model not in f.name or (a.variant != "all" and f".{a.variant}." not in f.name):
            continue
        is_asr = ".asr." in f.name
        size: dict[str, list[tuple[int, ...]]] = {g: [] for g, _ in GROUPS}
        for r in res:
            if not r["gold"]:
                continue
            s = samples[r["id"]]
            p = build_prompt(s, "train", asr[s["audio"]] if is_asr else s["audio_text"], asr if is_asr else None)
            msgs = [{"role": "system", "content": p["system"]}] + p["messages"]
            chars = len(p["system"]) + len(json.dumps(p["tools"], ensure_ascii=False)) + sum(len(json.dumps(m, ensure_ascii=False)) for m in p["messages"])
            toks = len(tok(tok.apply_chat_template(msgs, tools=p["tools"], add_generation_prompt=True, tokenize=False, enable_thinking=False))["input_ids"]) if tok else 0
            row = (toks, chars, len(p["tools"]), len({t["function"]["name"].split(".")[0] for t in p["tools"]}), sum(1 for h in s["history"] if h["role"] == "user"))
            size[group_of(r)].append(row)
            size["all"].append(row)
        print(f"\n== {v}: {f.name} ({len(size['all'])} samples whose gold has a tool call)")
        print(f"   {'group':38s} {'n':>3s} | {'tokens mean':>11s} {'median':>7s} | {'chars mean':>10s} {'median':>7s} | {'tools mean':>10s} {'median':>6s} | {'apps mean':>9s} {'median':>6s} | {'history turns mean':>18s}")
        for g, title in GROUPS:
            xs = size[g]
            if not xs:
                print(f"   {title:38s} {0:3d} |")
                continue
            col = lambda i: [x[i] for x in xs]  # noqa: E731
            tk = f"{st.mean(col(0)):11.0f} {st.median(col(0)):7.0f}" if tok else f"{'-':>11s} {'-':>7s}"
            print(f"   {title:38s} {len(xs):3d} | {tk} | {st.mean(col(1)):10.0f} {st.median(col(1)):7.0f} | {st.mean(col(2)):10.1f} {st.median(col(2)):6.0f} | {st.mean(col(3)):9.1f} {st.median(col(3)):6.0f} | {st.mean(col(4)):18.1f}")


if __name__ == "__main__":
    main()
