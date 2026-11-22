"""Compare two ASR outputs on the real train clips: base PhoWhisper vs the fine-tuned one (kaggle/asr_decode).

  .venv\\Scripts\\python scripts/asr_compare.py --base work/asr/phowhisper/train.jsonl --new work/asr/phowhisper_ft/train.jsonl

Reference text and fold come from work/kaggle_phowhisper_input_q/real.jsonl. Metrics as in scripts/asr_report.py
(same normalisation): WER (all clips), gold-arg recall and hallucination (final user turn of each sample), per fold
group (fold 0 = held out from ASR training; folds 1-4 = trained on). Prints a few fold-0 clips where the two differ most.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jiwer  # noqa: E402

from asr_report import HALLU, contains, gold_strings, norm  # noqa: E402
from lifemate.data import PROJECT, load_samples  # noqa: E402

REAL = PROJECT / "work" / "kaggle_phowhisper_input_q" / "real.jsonl"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def score(hyp: dict, clips: list[str], ref: dict, finals: dict) -> dict:
    edits = words = g_hit = g_n = hallu = n_final = 0
    for c in clips:
        r, h = norm(ref[c]), norm(hyp[c])
        w = jiwer.wer(r, h) if h else 1.0
        edits += w * len(r.split())
        words += len(r.split())
        if c in finals:
            n_final += 1
            gs = gold_strings(finals[c])
            g_n += len(gs)
            g_hit += sum(contains(hyp[c], g) for g in gs)
            hallu += bool(HALLU.search(hyp[c])) or w >= 0.8
    return {"clips": len(clips), "wer": edits / max(words, 1), "gold": g_hit / max(g_n, 1), "gold_n": g_n,
            "hallu": hallu / max(n_final, 1), "finals": n_final}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    real = jsonl(REAL)
    ref = {r["audio"]: r["text"] for r in real}
    fold = {r["audio"]: r["fold"] for r in real}
    finals = {s["audio"]: s for s in load_samples("train")}
    base = {r["audio"]: r["hyp"] for r in jsonl(a.base)}
    new = {r["audio"]: r["hyp"] for r in jsonl(a.new)}
    groups = {"fold 0 (held out)": [c for c in sorted(ref) if fold[c] == 0],
              "folds 1-4 (trained)": [c for c in sorted(ref) if fold[c] != 0]}
    print(f"{'slice':<22} {'model':<5} {'clips':>5} {'WER':>7} {'gold-arg recall':>16} {'hallu':>7}")
    for name, clips in groups.items():
        for tag, hyp in (("base", base), ("new", new)):
            have = [c for c in clips if c in hyp]
            if not have:
                print(f"{name:<22} {tag:<5}   (no hypotheses)")
                continue
            s = score(hyp, have, ref, finals)
            print(f"{name:<22} {tag:<5} {s['clips']:>5} {s['wer']:>7.4f} {s['gold']:>9.4f} ({s['gold_n']:>4}) {s['hallu']:>7.4f}")
    f0 = [c for c in groups["fold 0 (held out)"] if c in base and c in new]
    gap = sorted(f0, key=lambda c: (jiwer.wer(norm(ref[c]), norm(new[c]) or "x") - jiwer.wer(norm(ref[c]), norm(base[c]) or "x")))
    for title, picks in (("new better", gap[:a.show]), ("new worse", gap[::-1][:a.show])):
        print(f"\n=== fold 0, {title} ===")
        for c in picks:
            print(f"--- {c}\n    ref : {ref[c]}\n    base: {base[c]}\n    new : {new[c]}")


if __name__ == "__main__":
    main()
