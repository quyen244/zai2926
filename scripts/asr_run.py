"""Stage 2.1: PhoWhisper-large (CTranslate2) over every user audio clip (final + history turns) of the given splits.

  venv/bin/python scripts/asr_run.py --splits train public_test            # full run
  venv/bin/python scripts/asr_run.py --splits public_test --limit 20       # smoke

Writes work/asr/phowhisper/<split>.jsonl ({audio, hyp, redo, sec_audio, sec_decode, avg_logprob, no_speech_prob,
compression_ratio}), keyed by the audio path relative to the split. Resumable. Hallucination guard: a clip whose hypothesis
is empty / a known filler phrase / repetitive is re-decoded once with peak normalisation + Silero VAD, and the new text is
kept only if it is no longer flagged.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lifemate.cuda_dlls  # noqa: E402,F401  (before faster_whisper)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

from lifemate.data import PROJECT, audio_path, load_samples  # noqa: E402

MODEL = PROJECT / "work" / "models" / "phowhisper-large-ct2"
OUT = PROJECT / "work" / "asr" / "phowhisper"
HALLU = re.compile(r"subscribe|đăng k[ýí]|ghiền mì gõ|cảm ơn các bạn đã (theo dõi|xem)|hẹn gặp lại|like|share|la la school", re.I)


def clips(split: str) -> list[str]:
    """Relative audio paths of the final turn and all user history turns, sorted and unique."""
    out = set()
    for s in load_samples(split):
        out.add(s["audio"])
        out.update(t["audio"] for t in s.get("history") or [] if t.get("role") == "user" and t.get("audio"))
    return sorted(out)


def flagged(hyp: str, comp: float) -> bool:
    return not hyp.strip() or bool(HALLU.search(hyp)) or comp > 2.4


def peak_norm(x: np.ndarray, db: float = -3.0) -> np.ndarray:
    p = float(np.abs(x).max())
    return x * (10 ** (db / 20) / p) if p > 0 else x


def decode(fn, audio, **kw) -> tuple[str, dict]:
    segs, _ = fn(audio, language="vi", beam_size=5, condition_on_previous_text=False, **kw)
    segs = list(segs)
    return " ".join(s.text.strip() for s in segs).strip(), {
        "avg_logprob": round(min((s.avg_logprob for s in segs), default=0.0), 3),
        "no_speech_prob": round(max((s.no_speech_prob for s in segs), default=0.0), 3),
        "compression_ratio": round(max((s.compression_ratio for s in segs), default=0.0), 3)}


def run(split: str, model, limit: int | None) -> None:
    out = OUT / f"{split}.jsonl"
    done = set()
    if out.is_file():
        done = {json.loads(l)["audio"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [c for c in clips(split) if c not in done][:limit]
    print(f"{split}: {len(done)} done, {len(todo)} to go", flush=True)
    t_all = time.time()
    audio_sec = 0.0
    with open(out, "a", encoding="utf-8") as f:
        for k, rel in enumerate(todo, 1):
            x, sr = sf.read(audio_path(split, rel), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            t = time.time()
            hyp, m = decode(model.transcribe, x)
            redo = False
            if flagged(hyp, m["compression_ratio"]):
                h2, m2 = decode(model.transcribe, peak_norm(x), vad_filter=True)
                if h2 and not flagged(h2, m2["compression_ratio"]):
                    hyp, m, redo = h2, m2, True
            f.write(json.dumps({"audio": rel, "hyp": hyp, "redo": redo, "sec_audio": round(len(x) / sr, 2),
                                "sec_decode": round(time.time() - t, 2), **m}, ensure_ascii=False) + "\n")
            audio_sec += len(x) / sr
            if k % 100 == 0:
                print(f"{split}: {k}/{len(todo)}  RTF {(time.time() - t_all) / audio_sec:.3f}", flush=True)
    if todo:
        print(f"{split}: {len(todo)} clips, {audio_sec:.0f}s audio in {time.time() - t_all:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "public_test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compute", default="float16")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(str(MODEL), device="cuda", compute_type=a.compute)
    for sp in a.splits:
        run(sp, model, a.limit)


if __name__ == "__main__":
    main()
