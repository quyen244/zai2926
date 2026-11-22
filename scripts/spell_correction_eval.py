"""Paired WER evaluation of nrl-ai Vietnamese spell correction on PhoWhisper FT output.

Uses CV fold 0 only, keeping every evaluated utterance out of any later SFT on folds 1-4.
Writes raw hypotheses and aggregate metrics under work/spell_correction/.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jiwer
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lifemate.data import load_samples

MODEL_ID = "nrl-ai/vn-spell-correction-small"
NUM = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(NUM.get(w, w) for w in s.split())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    asr_path = ROOT / "work/asr/phowhisper_ft/train.jsonl"
    asr = {r["audio"]: r["hyp"] for r in map(json.loads, asr_path.read_text(encoding="utf-8").splitlines()) if r.get("audio")}
    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    samples = [s for s in load_samples("train") if cv[s["id"]] == 0 and s["audio"] in asr]
    if a.limit:
        samples = samples[:a.limit]
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} fold0_samples={len(samples)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    out = []
    with torch.inference_mode():
        for start in range(0, len(samples), a.batch_size):
            batch = samples[start:start + a.batch_size]
            inputs = tok([asr[s["audio"]] for s in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            ids = model.generate(**inputs, num_beams=4, max_new_tokens=192)
            hyps = tok.batch_decode(ids, skip_special_tokens=True)
            for sample, corrected in zip(batch, hyps):
                hyp = asr[sample["audio"]]
                ref_n, hyp_n, cor_n = norm(sample["audio_text"]), norm(hyp), norm(corrected)
                out.append({"id": sample["id"], "ref": sample["audio_text"], "asr": hyp, "corrected": corrected,
                            "ref_words": len(ref_n.split()), "asr_edits": jiwer.wer(ref_n, hyp_n) * len(ref_n.split()),
                            "corrected_edits": jiwer.wer(ref_n, cor_n) * len(ref_n.split())})
            print(f"{min(start + len(batch), len(samples))}/{len(samples)}", flush=True)
    nwords = sum(r["ref_words"] for r in out)
    result = {"model": MODEL_ID, "split": "cv5 fold 0", "n": len(out), "ref_words": nwords,
              "wer_before": sum(r["asr_edits"] for r in out) / nwords,
              "wer_after": sum(r["corrected_edits"] for r in out) / nwords,
              "delta": (sum(r["corrected_edits"] for r in out) - sum(r["asr_edits"] for r in out)) / nwords}
    dest = ROOT / "work/spell_correction"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fold0.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (dest / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
