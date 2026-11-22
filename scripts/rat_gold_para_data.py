"""Build RAT tool-only SFT data from gold text and accepted gold paraphrases only.

The existing gold rows already have checked tool-choice rationales and gold calls.
Each paraphrase changes only the final user utterance. No ASR or recovery row is used.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lifemate.data import load_samples  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from lifemate.sft import encode  # noqa: E402
from recovery_data import body_start  # noqa: E402
import rationale_data as rationale  # noqa: E402

SOURCE = ROOT / "work/sft_rat_toolonly"
PARAPHRASES = ROOT / "work/synth/paraphrase.kept.jsonl"
OUT = ROOT / "work/sft_rat_toolonly_gold_para"
MAX_LEN = 10240


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def calls(row: dict) -> list[dict]:
    result = []
    for message in row["messages"][body_start(row["messages"]):]:
        if not message.get("train"):
            continue
        for call in message.get("tool_calls") or []:
            func = call["function"]
            args = func["arguments"]
            result.append({"name": func["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
    return result


def check_row(row: dict, sample: dict) -> None:
    if row["variant"] not in {"gold", "para1", "para2"}:
        raise ValueError(f"unexpected variant: {row['variant']}")
    if any(message.get("train") is False for message in row["messages"]):
        raise ValueError(f"recovery message in {row['id']}")
    if calls(row) != sample["label"]["tool_calls"]:
        raise ValueError(f"tool call or arguments differ from gold: {row['id']} {row['variant']}")
    for message in row["messages"]:
        if message.get("train") and message.get("tool_calls"):
            content = message.get("content") or ""
            if "<select>" not in content or "<tool>" not in content or "<args>" in content:
                raise ValueError(f"invalid tool-only rationale: {row['id']} {row['variant']}")


def main() -> None:
    train = jsonl(SOURCE / "traj.train.jsonl")
    val = jsonl(SOURCE / "traj.val.jsonl")
    gold = {row["id"]: row for row in train if row["variant"] == "gold"}
    if len(gold) != 1115 or len(val) != 59 or any(row["variant"] != "gold" for row in val):
        raise ValueError("source RAT gold/val data changed")
    samples = {sample["id"]: sample for sample in load_samples("train")}
    val_ids = {row["id"] for row in val}
    replies = rationale.load("gold")
    paraphrases = jsonl(PARAPHRASES)
    if len(paraphrases) != 2115:
        raise ValueError(f"accepted paraphrase count changed: {len(paraphrases)}")
    seen = set()
    rows = list(gold.values())
    dropped = Counter()
    rationale_drop_reasons = Counter()
    for item in paraphrases:
        sid, number, text = item["id"], item["n"], item["text"]
        key = (sid, number)
        if key in seen or number not in (1, 2) or not text.strip() or not item.get("phash"):
            raise ValueError(f"invalid accepted paraphrase: {key}")
        seen.add(key)
        if sid in val_ids:
            raise ValueError(f"paraphrase leaked into val: {sid}")
        if sid not in gold:
            dropped["no_checked_gold_rationale"] += 1
            continue
        sample = samples[sid]
        para_sample = {**sample, "audio_text": text}
        if sample["label"]["tool_calls"]:
            reply = replies.get(sid)
            results = rationale.check(para_sample, rationale.steps_for(para_sample, "gold"), reply.get("content", ""), "gold") if reply else []
            problems = [bad for result in results for bad in result["bad"]]
            if reply is None or problems:
                dropped["rationale_inconsistent_with_paraphrase"] += 1
                rationale_drop_reasons.update(problem.split(" ", 1)[0] for problem in problems)
                continue
        prompt = build_prompt(para_sample, "train", text, None, normalise=False)
        original = gold[sid]
        row = {**original, "variant": f"para{number}", "system": prompt["system"], "tools": prompt["tools"],
               "messages": prompt["messages"] + original["messages"][body_start(original["messages"]):]}
        check_row(row, sample)
        rows.append(row)
    for row in rows + val:
        check_row(row, samples[row["id"]])
    if {row["id"] for row in rows} & val_ids:
        raise ValueError("train/val overlap")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    for row in rows + val:
        encoded = encode(row, tokenizer, MAX_LEN)
        if encoded["n_tokens"] > MAX_LEN:
            raise ValueError(f"row exceeds context: {row['id']} {row['variant']}")
    rows.sort(key=lambda row: (row["id"], row["variant"]))
    OUT.mkdir(parents=True, exist_ok=True)
    for split, items in (("train", rows), ("val", val)):
        (OUT / f"traj.{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8")
    summary = {"train_rows": len(rows), "val_rows": len(val), "variants": dict(Counter(row["variant"] for row in rows)),
               "accepted_paraphrases_input": len(paraphrases), "dropped": dict(dropped),
               "rationale_drop_reasons": dict(rationale_drop_reasons), "asr_rows": 0, "recovery_rows": 0}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
