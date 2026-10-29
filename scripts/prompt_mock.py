"""Stage 2.3: print the exact prompt lifemate.prompt.build_prompt would send for one sample, labelled
SYSTEM / TOOLS / MESSAGES, so the structure can be reviewed before any LLM is called.

  python scripts/prompt_mock.py --split train                                   # first sample with history
  python scripts/prompt_mock.py --split train --id train_0008
  python scripts/prompt_mock.py --split public_test --asr phowhisper/public_test  # ASR text instead of gold

Output contains real per-sample competition data (transcripts, profile, tool results): read it on screen
or redirect to a file under work/ (git-ignored). Never paste it into a report or any other git-tracked file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402


def load_asr(stem: str) -> dict[str, str]:
    path = PROJECT / "work" / "asr" / f"{stem}.jsonl"
    return {json.loads(line)["audio"]: json.loads(line)["hyp"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--id", help="sample id; default: first sample with non-empty history")
    ap.add_argument("--asr", help="work/asr/<stem>.jsonl to use as the text source (required for public_test)")
    a = ap.parse_args()

    samples = load_samples(a.split)
    sample = next(s for s in samples if s["id"] == a.id) if a.id else next((s for s in samples if s["history"]), samples[0])
    asr = load_asr(a.asr) if a.asr else {}
    user_text = asr.get(sample["audio"], sample.get("audio_text"))
    if user_text is None:
        raise SystemExit(f"{sample['id']} has no gold audio_text and --asr was not given")

    prompt = build_prompt(sample, a.split, user_text, asr=asr)

    label_n = len(sample["label"]["tool_calls"]) if "label" in sample else "n/a (no gold label)"
    print(f"SAMPLE {sample['id']}  split={a.split}  history_turns={len(sample['history'])}  label_tool_calls={label_n}")
    print("\n=== SYSTEM ===")
    print(prompt["system"])
    print(f"\n=== TOOLS ({len(prompt['tools'])}) ===")
    for t in prompt["tools"]:
        f = t["function"]
        print(f"- {f['name']}({', '.join((f['parameters'].get('properties') or {}).keys())}): {f['description']}")
    print(f"\n=== MESSAGES ({len(prompt['messages'])}) ===")
    for m in prompt["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m["tool_calls"])
            print(f"[assistant -> tool_calls] {calls}")
        elif m["role"] == "tool":
            print(f"[tool result] {m['content']}")
        else:
            print(f"[{m['role']}] {m['content']}")

    chars = len(prompt["system"]) + sum(len(json.dumps(t, ensure_ascii=False)) for t in prompt["tools"]) + \
        sum(len(json.dumps(m, ensure_ascii=False)) for m in prompt["messages"])
    print(f"\n~{chars:,} chars, ~{chars // 3:,} tokens (chars/3 estimate for Vietnamese)")


if __name__ == "__main__":
    main()
