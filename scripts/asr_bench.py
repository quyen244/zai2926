"""Stage 1.2: transcribe a stratified subset of train final-turn clips with several ASR models (faster-whisper / CTranslate2).

  .venv\\Scripts\\python scripts/asr_bench.py --models large-v3 large-v3-turbo work/models/phowhisper-large-ct2 --n 400

Writes work/asr/bench_ids.json and work/asr/<model tag>.jsonl ({id, hyp, sec_audio, sec_decode, avg_logprob,
no_speech_prob, compression_ratio}). Resumable: ids already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lifemate.cuda_dlls  # noqa: E402,F401  (before faster_whisper)
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

from lifemate.data import PROJECT, audio_path, load_samples  # noqa: E402
from lifemate.splits import label_type  # noqa: E402

OUT = PROJECT / "work" / "asr"


def bench_ids(n: int, seed: int = 7) -> list[str]:
    path = OUT / "bench_ids.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    samples = load_samples("train")
    strata = defaultdict(list)
    for s in samples:
        with sf.SoundFile(audio_path("train", s["audio"])) as f:
            dur = f.frames / f.samplerate
        strata[(label_type(s["label"]["tool_calls"]), "short" if dur < 5 else "mid" if dur < 9 else "long")].append(s["id"])
    rng = random.Random(seed)
    ids = []
    total = len(samples)
    for key in sorted(strata):
        group = sorted(strata[key])
        rng.shuffle(group)
        ids += group[:max(1, round(n * len(group) / total))]
    ids = sorted(ids)
    path.write_text(json.dumps(ids), encoding="utf-8")
    return ids


def tag(model: str) -> str:
    return Path(model).name if ("/" in model or "\\" in model) else model


def run(model: str, ids: list[str], by_id: dict) -> None:
    out = OUT / f"{tag(model)}.jsonl"
    done = set()
    if out.is_file():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [i for i in ids if i not in done]
    if not todo:
        print(f"{tag(model)}: complete")
        return
    t0 = time.time()
    m = WhisperModel(model, device="cuda", compute_type="int8_float16")
    print(f"{tag(model)}: loaded in {time.time() - t0:.0f}s, {len(todo)} clips to go", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for k, sid in enumerate(todo, 1):
            audio, sr = sf.read(audio_path("train", by_id[sid]["audio"]), dtype="float32")
            t = time.time()
            segs, _ = m.transcribe(audio, language="vi", beam_size=5, condition_on_previous_text=False)
            segs = list(segs)
            row = {"id": sid, "hyp": " ".join(s.text.strip() for s in segs).strip(),
                   "sec_audio": round(len(audio) / sr, 2), "sec_decode": round(time.time() - t, 2),
                   "avg_logprob": round(min((s.avg_logprob for s in segs), default=0.0), 3),
                   "no_speech_prob": round(max((s.no_speech_prob for s in segs), default=0.0), 3),
                   "compression_ratio": round(max((s.compression_ratio for s in segs), default=0.0), 3)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"{tag(model)}: {k}/{len(todo)}", flush=True)
    del m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {s["id"]: s for s in load_samples("train")}
    ids = bench_ids(a.n)
    print(f"{len(ids)} bench clips", flush=True)
    for model in a.models:
        run(model, ids, by_id)


if __name__ == "__main__":
    main()
