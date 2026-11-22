from __future__ import annotations

import json
import os
import random
import time
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from vieneu import Vieneu

ROOT = Path(os.environ["ASR_DATA"])
OUT = Path(os.environ["ASR_OUT"])
SHARD = int(os.environ["SHARD_INDEX"])
SHARDS = int(os.environ["SHARD_COUNT"])
DIALECTS = {
    "north": ["Minh Đức", "Phạm Tuyên", "Xuân Vĩnh", "Thanh Bình", "Ngọc Linh", "Đoan Trang", "Quỳnh Anh", "Quốc Tuấn"],
    "central": ["Quang Sơn", "Ngọc Trân"],
    "south": ["Adam", "Thái Sơn", "Thục Đoan", "Minh Triết", "Mỹ Duyên", "Đức Trí", "Kim Thanh", "Thùy Dung"],
}


def listening_profile() -> tuple[list[tuple[str, float]], float, float, int]:
    with (ROOT / "listening.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    annotated = [r for r in rows if any((r.get(k) or "").strip() for k in ("noise", "volume", "speed", "clarity", "hard_words", "notes"))]
    n = max(1, len(annotated))
    noise = sum((r.get("noise") or "").strip() == "1 some" for r in annotated) / n
    hard = sum((r.get("clarity") or "").strip() in {"hard", "unintelligible"} for r in annotated) / n
    nonnormal_volume = sum((r.get("volume") or "").strip() in {"quiet", "loud"} for r in annotated) / n
    nonnormal_speed = sum((r.get("speed") or "").strip() in {"fast", "slow"} for r in annotated) / n
    weights = [("speed", max(0.1, nonnormal_speed)), ("pitch", 0.18), ("gain", max(0.08, nonnormal_volume)),
               ("noise", max(0.08, noise)), ("local_noise", max(0.10, hard * 0.55)), ("reverb", max(0.08, hard * 0.35))]
    return weights, noise, hard, n


def augment(x: np.ndarray, sr: int, rng: random.Random, profile) -> tuple[np.ndarray, list[str]]:
    import librosa
    applied = []
    weights = profile[0]
    count = rng.choices([1, 2, 3], [0.62, 0.32, 0.06])[0]
    ops = []
    available = list(weights)
    while available and len(ops) < count:
        op = rng.choices([x[0] for x in available], [x[1] for x in available])[0]
        ops.append(op)
        available = [x for x in available if x[0] != op]
    for op in ops:
        if op == "speed":
            rate = rng.uniform(0.90, 1.10)
            x = librosa.effects.time_stretch(x, rate=rate)
            applied.append(f"speed:{rate:.2f}")
        elif op == "pitch":
            steps = rng.uniform(-1.25, 1.25)
            x = librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)
            applied.append(f"pitch:{steps:.2f}")
        elif op == "gain":
            gain = rng.uniform(0.72, 1.28)
            x *= gain
            applied.append(f"gain:{gain:.2f}")
        elif op == "noise":
            snr = rng.uniform(12, 28)
            rms = float(np.sqrt(np.mean(x*x) + 1e-12))
            noise = rng.random() * np.random.default_rng(rng.randrange(2**32)).normal(size=len(x)).astype(np.float32)
            nrms = float(np.sqrt(np.mean(noise*noise) + 1e-12))
            x += noise * (rms / (nrms * (10 ** (snr / 20))))
            applied.append(f"noise_snr:{snr:.1f}")
        elif op == "local_noise":
            if len(x) > sr // 2:
                start = rng.randrange(0, max(1, len(x) - sr // 5))
                length = min(rng.randrange(sr // 8, sr // 3), len(x) - start)
                noise = np.random.default_rng(rng.randrange(2**32)).normal(size=length).astype(np.float32)
                rms = float(np.sqrt(np.mean(x*x) + 1e-12))
                x[start:start+length] += noise * (rms * rng.uniform(0.04, 0.12))
                applied.append(f"local_noise:{start/sr:.2f}:{length/sr:.2f}")
        elif op == "reverb":
            ir_len = max(8, int(sr * rng.uniform(0.08, 0.22)))
            t = np.arange(ir_len, dtype=np.float32) / sr
            ir = np.exp(-t * rng.uniform(12, 24)) * np.random.default_rng(rng.randrange(2**32)).normal(size=ir_len)
            ir[0] += 1.0
            # Preserve the direct-path level. L1-normalising the whole impulse response
            # attenuates speech by about 25-35 dB and creates nearly inaudible training clips.
            ir /= abs(float(ir[0])) + 1e-12
            x = np.convolve(x, ir.astype(np.float32), mode="full")[:len(x)]
            applied.append("reverb")
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x.astype(np.float32), applied


def main():
    tts = Vieneu()
    profile = listening_profile()
    if SHARD == 0:
        print(f"listening_profile annotated={profile[3]} noise={profile[1]:.2f} hard={profile[2]:.2f} weights={profile[0]}", flush=True)
    rows = [json.loads(l) for l in (ROOT / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT / f"audio_{SHARD}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / f"shard_{SHARD}.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(l)["item_id"] for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    selected = [r for i, r in enumerate(rows) if i % SHARDS == SHARD]
    started = time.time()
    n = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for row in selected:
            for ri, text in enumerate(row["rollouts"]):
                for dialect, voices in DIALECTS.items():
                    item_id = f"{row['id']}_r{ri+1}_{dialect}"
                    if item_id in done:
                        continue
                    rng = random.Random(f"{row['id']}:{ri}:{dialect}:2026")
                    voice = rng.choice(voices)
                    audio = np.asarray(tts.infer(text, voice=voice), dtype=np.float32).reshape(-1)
                    audio = resample_poly(audio, 1, 3).astype(np.float32)  # 48 kHz -> 16 kHz
                    audio, aug = augment(audio, 16000, rng, profile)
                    fn = item_id + ".wav"
                    sf.write(audio_dir / fn, audio, 16000, subtype="PCM_16")
                    record = {"item_id": item_id, "source_id": row["id"], "rollout": ri+1,
                              "dialect": dialect, "voice": voice, "text": text,
                              "audio": f"audio_{SHARD}/{fn}", "sample_rate": 16000,
                              "augmentations": aug}
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    mf.flush()
                    done.add(item_id)
                    n += 1
                    if n % 30 == 0:
                        print(f"gpu={SHARD} done={n} shard_total={len(selected)*9} elapsed={time.time()-started:.1f}s", flush=True)
    print(f"SHARD_DONE gpu={SHARD} count={n} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
