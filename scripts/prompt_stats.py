"""How big is the agent prompt: tools per prompt, apps per prompt, characters and (optionally) tokens.

  python scripts/prompt_stats.py                         # train and public_test, tool and app counts, characters
  python scripts/prompt_stats.py --split train --tokens  # also Qwen3 tokens after the chat template (needs the tokenizer on disk)
  python scripts/prompt_stats.py --id train_0018 --tokens   # one or more samples: every tool name in its prompt, grouped by app

The prompt is built exactly as the agent builds it (lifemate.prompt.build_prompt: reference text on train; on other
splits an empty final turn and the cached PhoWhisper text for history turns, since only train has transcripts. The
user text is a few dozen tokens, the tools and history dominate). Reads local files only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402


def describe(name: str, xs: list[float]) -> str:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return (f"{name:28s} mean {st.mean(s):8.1f} | median {st.median(s):8.1f} | min {s[0]:7.0f} | p90 {q(0.9):7.0f}"
            f" | p99 {q(0.99):7.0f} | max {s[-1]:7.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "public_test"])
    ap.add_argument("--tokens", action="store_true", help="count Qwen3 tokens of the rendered prompt (slower)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--id", nargs="+", help="per-sample view of these ids (split taken from the id prefix)")
    a = ap.parse_args()

    tok = None
    if a.tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    if a.id:
        a.split = sorted({i.rsplit("_", 1)[0] for i in a.id})
    for split in a.split:
        samples = load_samples(split)
        if a.id:
            samples = [s for s in samples if s["id"] in a.id]
        asr_file = PROJECT / "work" / "asr" / "phowhisper" / f"{split}.jsonl"  # history turns of non-train splits are audio only
        asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in asr_file.read_text(encoding="utf-8").splitlines() if l.strip()} if asr_file.is_file() else None
        tools, apps, chars, tool_chars, toks = [], [], [], [], []
        for s in samples:
            p = build_prompt(s, split, s.get("audio_text") or "", asr if split != "train" else None)
            tools.append(len(p["tools"]))
            apps.append(len({t["function"]["name"].split(".")[0] for t in p["tools"]}))
            tj = json.dumps(p["tools"], ensure_ascii=False)
            tool_chars.append(len(tj))
            chars.append(len(p["system"]) + len(tj) + sum(len(json.dumps(m, ensure_ascii=False)) for m in p["messages"]))
            if tok is not None:
                text = tok.apply_chat_template([{"role": "system", "content": p["system"]}] + p["messages"], tools=p["tools"],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                toks.append(len(tok(text)["input_ids"]))
            if a.id:
                by_app = {}
                for t in p["tools"]:
                    by_app.setdefault(t["function"]["name"].split(".")[0], []).append(t["function"]["name"].split(".", 1)[1])
                gold = [c["name"] for c in (s.get("label") or {}).get("tool_calls", [])]
                print(f"== {s['id']}: {tools[-1]} tools, {apps[-1]} apps, {chars[-1]} chars" + (f", {toks[-1]} Qwen3 tokens" if toks else ""))
                print(f"   gold calls: {gold or '[]'}")
                for app, fs in sorted(by_app.items()):
                    print(f"   {app:14s} {len(fs):2d}  {', '.join(fs)}")
        if a.id:
            continue
        print(f"== {split}: {len(samples)} samples (tools include sys.ask_missing_info) ==")
        print(describe("tools per prompt", tools))
        print(describe("apps per prompt", apps))
        print(describe("chars, tool schemas only", tool_chars))
        print(describe("chars, whole prompt", chars))
        if toks:
            print(describe("Qwen3 tokens, whole prompt", toks))
            print(f"{'prompts over 16384 tokens':28s} {sum(t > 16384 for t in toks)}")


if __name__ == "__main__":
    main()
