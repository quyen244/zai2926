"""Audit held-out PhoWhisper P2 errors using the project's WER normalization."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from asr_report import norm  # noqa: E402

REAL = ROOT / "work/kaggle_phowhisper_base_tquyen/real.jsonl"
DECODE = ROOT / "work/kaggle/asr_decode/out/new.jsonl"
REPORT = ROOT / "work/asr_p2_audit.md"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    refs = [r for r in jsonl(REAL) if r["fold"] == 0]
    hyps = {r["audio"]: r for r in jsonl(DECODE)}
    assert len(refs) == 1009 and all(r["audio"] in hyps for r in refs)
    total_edits = total_words = perfect = severe = 0
    substitutions = deletions = insertions = 0
    final_edits = final_words = final_clips = 0
    edits = Counter()
    examples = []
    for row in refs:
        ref, hyp = norm(row["text"]), norm(hyps[row["audio"]]["hyp"])
        score = jiwer.process_words(ref, hyp)
        total_edits += score.substitutions + score.deletions + score.insertions
        substitutions += score.substitutions
        deletions += score.deletions
        insertions += score.insertions
        total_words += len(ref.split())
        if "_u" not in row["id"]:
            final_clips += 1
            final_edits += score.substitutions + score.deletions + score.insertions
            final_words += len(ref.split())
        perfect += score.wer == 0
        severe += score.wer >= 0.5
        examples.append((score.wer, row, hyps[row["audio"]]))
        rwords, hwords = ref.split(), hyp.split()
        for chunk in score.alignments[0]:
            if chunk.type == "substitute" and chunk.ref_end_idx - chunk.ref_start_idx == 1 and chunk.hyp_end_idx - chunk.hyp_start_idx == 1:
                edits[(rwords[chunk.ref_start_idx], hwords[chunk.hyp_start_idx])] += 1
    lines = [
        "# PhoWhisper P2 fold0 error audit",
        "",
        f"- Clips: {len(refs)}; reference words: {total_words}; edit operations: {total_edits}; WER: {total_edits / total_words:.4%}.",
        f"- Error types: {substitutions} substitutions, {deletions} deletions, {insertions} insertions.",
        f"- Final user turns: {final_clips} clips; {final_edits}/{final_words} edits/words; WER: {final_edits / final_words:.4%}.",
        f"- Perfect transcripts: {perfect}; WER at least 50%: {severe}.",
        "- Normalization: lowercase, NFC, remove punctuation, convert single number words zero through ten to digits.",
        "- Full-utterance WER can count different number conventions as errors, such as `24` and `hai bốn`; this needs a separate audited metric before changing production text.",
        "",
        "## Frequent one-word substitutions",
        "",
    ]
    lines += [f"- `{a}` -> `{b}`: {n}" for (a, b), n in edits.most_common(25)]
    lines += ["", "## Highest-WER examples", ""]
    for score, row, hyp in sorted(examples, key=lambda x: x[0], reverse=True)[:20]:
        lines += [f"- `{row['id']}`: WER {score:.1%}", f"  - Gold: {row['text']}", f"  - P2: {hyp['hyp']}"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}; WER {total_edits / total_words:.4%}; perfect {perfect}; severe {severe}")


if __name__ == "__main__":
    main()
