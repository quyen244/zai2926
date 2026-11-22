"""Print paired fold-0 gold, raw P2, and Qwen few-shot correction examples.

  .venv\\Scripts\\python scripts/show_asr_correction_samples.py --n 10
  .venv\\Scripts\\python scripts/show_asr_correction_samples.py --n 10 --seed 17

The script reads the competition transcript cache and local correction outputs, prints per-sample WER,
and writes nothing. Per-sample content must remain out of version control.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_report import norm  # noqa: E402

import jiwer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "work/kaggle_phowhisper_input_q/real.jsonl"
BASELINE = ROOT / "work/kaggle/asr_p2_decode_sweep_tquyen/out/baseline.jsonl"
PREDICTIONS = ROOT / "work/kaggle/asr_fewshot_correction_output/out/predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wer_counts(reference: str, hypothesis: str) -> tuple[float, int, int]:
    result = jiwer.process_words(norm(reference), norm(hypothesis))
    edits = result.substitutions + result.deletions + result.insertions
    return result.wer, edits, len(norm(reference).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of fold-0 examples")
    parser.add_argument("--seed", type=int, default=17, help="fixed random sample seed")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    sys.stdout.reconfigure(encoding="utf-8")

    real = {row["audio"]: row for row in read_jsonl(REAL) if row["fold"] == 0}
    baseline = {row["audio"]: row["hyp"] for row in read_jsonl(BASELINE)}
    predictions: dict[tuple[str, int], dict] = {}
    for row in read_jsonl(PREDICTIONS):
        key = row["audio"], int(row["k"])
        if key in predictions:
            raise ValueError(f"duplicate correction row: {key}")
        predictions[key] = row

    if len(real) != 1009 or len(baseline) != 1009:
        raise ValueError(f"expected 1,009 fold-0 references and raw P2 rows; got {len(real)} and {len(baseline)}")
    if any((audio, k) not in predictions for audio in real for k in (2, 3)):
        raise ValueError("correction output is missing a fold-0 audio/k pair")

    audio_ids = sorted(real)
    selected = sorted(random.Random(args.seed).sample(audio_ids, min(args.n, len(audio_ids))))
    print(f"Showing {len(selected)} / {len(audio_ids)} fold-0 clips (seed={args.seed}); WER uses scripts/asr_report.py normalization.\n")
    total_words = 0
    total_edits = {"raw": 0, 2: 0, 3: 0}
    for audio in selected:
        ref = real[audio]["text"]
        raw = baseline[audio]
        raw_wer, raw_edits, words = wer_counts(ref, raw)
        c2 = predictions[(audio, 2)]["corrected"]
        c3 = predictions[(audio, 3)]["corrected"]
        w2, e2, _ = wer_counts(ref, c2)
        w3, e3, _ = wer_counts(ref, c3)
        total_words += words
        total_edits["raw"] += raw_edits
        total_edits[2] += e2
        total_edits[3] += e3
        print(f"=== {audio} | {words} reference words ===")
        print(f"GOLD      : {ref}")
        print(f"ASR P2    : {raw}  [WER {raw_wer:.1%}]")
        print(f"Qwen top-2: {c2}  [WER {w2:.1%}]")
        print(f"Qwen top-3: {c3}  [WER {w3:.1%}]\n")

    if total_words:
        print("=== Selected-sample aggregate ===")
        print(f"Raw P2    : {total_edits['raw']}/{total_words} edits, {total_edits['raw'] / total_words:.2%} WER")
        for k in (2, 3):
            print(f"Qwen top-{k}: {total_edits[k]}/{total_words} edits, {total_edits[k] / total_words:.2%} WER")


if __name__ == "__main__":
    main()
