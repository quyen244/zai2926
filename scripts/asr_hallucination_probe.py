"""Stage 1.2 follow-up: do Whisper's whole-clip hallucinations correlate with simple signal properties?

Compares hallucinated clips (known filler phrase or WER >= 0.8) with the rest of the bench set on: peak and RMS level
(dBFS), leading/trailing silence, clipping, duration. Then re-decodes the hallucinated clips with two cheap fixes
(peak normalisation to -3 dBFS, Silero VAD) to see which one removes the hallucination.
Run: .venv\\Scripts\\python scripts/asr_hallucination_probe.py --model large-v3
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lifemate.cuda_dlls  # noqa: E402,F401
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lifemate.data import PROJECT, audio_path, load_samples  # noqa: E402

HALLU = re.compile(r"subscribe|đăng k[ýí]|ghiền mì gõ|cảm ơn các bạn đã (theo dõi|xem)|hẹn gặp lại|like|share|la la school", re.I)


def stats(x: np.ndarray, sr: int) -> dict:
    eps = 1e-9
    frame = int(0.02 * sr)
    n = len(x) // frame
    e = 20 * np.log10(np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1)) + eps)
    voiced = np.where(e > e.max() - 35)[0]
    return {"peak_db": float(20 * np.log10(np.abs(x).max() + eps)), "rms_db": float(20 * np.log10(np.sqrt((x ** 2).mean()) + eps)),
            "lead_sil": float(voiced[0] * 0.02) if len(voiced) else 0.0, "trail_sil": float((n - 1 - voiced[-1]) * 0.02) if len(voiced) else 0.0,
            "clip_frac": float((np.abs(x) > 0.99).mean()), "dur": len(x) / sr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    a = ap.parse_args()
    train = {s["id"]: s for s in load_samples("train")}
    rows = [json.loads(line) for line in (PROJECT / "work" / "asr" / f"{a.model}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hallu_ids = {r["id"] for r in rows if HALLU.search(r["hyp"]) or not r["hyp"].strip()}
    feats = {}
    for r in rows:
        x, sr = sf.read(audio_path("train", train[r["id"]]["audio"]), dtype="float32")
        feats[r["id"]] = stats(x, sr)
    print(f"{len(hallu_ids)} hallucinated / {len(rows)} clips ({a.model})\n")
    print(f"{'feature':12} {'hallucinated (median)':>22} {'others (median)':>16}")
    for k in ("peak_db", "rms_db", "lead_sil", "trail_sil", "clip_frac", "dur"):
        h = statistics.median(feats[i][k] for i in hallu_ids) if hallu_ids else float("nan")
        o = statistics.median(feats[i][k] for i in feats if i not in hallu_ids)
        print(f"{k:12} {h:22.3f} {o:16.3f}")

    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device="cuda", compute_type="int8_float16")
    fixed = {"peak_norm": 0, "vad": 0, "both": 0}
    for sid in sorted(hallu_ids):
        x, sr = sf.read(audio_path("train", train[sid]["audio"]), dtype="float32")
        xn = x * (10 ** (-3 / 20) / max(np.abs(x).max(), 1e-6))
        for name, audio, vad in (("peak_norm", xn, False), ("vad", x, True), ("both", xn, True)):
            segs, _ = m.transcribe(audio, language="vi", beam_size=5, condition_on_previous_text=False, vad_filter=vad)
            hyp = " ".join(s.text.strip() for s in segs)
            ok = bool(hyp.strip()) and not HALLU.search(hyp)
            fixed[name] += ok
    print(f"\nre-decode of the {len(hallu_ids)} hallucinated clips, hallucination removed: {fixed}")


if __name__ == "__main__":
    main()
