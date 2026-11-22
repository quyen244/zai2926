from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = next(Path("/kaggle/input").glob("**/rollouts.jsonl")).parent


def main():
    started = time.time()
    train = DATA / "worker.py"
    if not train.is_file():
        raise SystemExit(f"worker.py missing from input dataset: {DATA}")
    print("GPU INVENTORY", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "vieneu", "librosa", "soundfile", "scipy"], check=True)
    out = WORK / "out"
    procs = []
    for shard in range(2):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard), "ASR_DATA": str(DATA),
               "ASR_OUT": str(out), "SHARD_INDEX": str(shard), "SHARD_COUNT": "2"}
        log_path = WORK / f"worker_{shard}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, str(train)], env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        procs.append((p, log_handle, log_path))
        print(f"started shard={shard} on GPU={shard}", flush=True)
    rc = 0
    for shard, (p, handle, log_path) in enumerate(procs):
        code = p.wait()
        handle.close()
        print(f"worker={shard} exit={code}", flush=True)
        if code:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            rc = code
    if rc:
        raise SystemExit(rc)
    records = []
    for shard in range(2):
        path = out / f"shard_{shard}.jsonl"
        if path.exists():
            records.extend(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    merged = out / "manifest.jsonl"
    merged.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    stats = {"sources": len({r["source_id"] for r in records}), "records": len(records),
             "expected_records": sum(3 * len(json.loads(l)["rollouts"]) for l in (DATA / "rollouts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()),
             "audio_files": sum(1 for _ in out.glob("audio_*/*.wav")), "runtime_seconds": time.time()-started,
             "gpu_workers": 2}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    archive = out / "audio.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for wav in sorted(out.glob("audio_*/*.wav")):
            zf.write(wav, wav.relative_to(out).as_posix())
    for shard in range(2):
        shutil.rmtree(out / f"audio_{shard}", ignore_errors=True)
    (out / "README.md").write_text(
        "ASR synthetic data: extract audio.zip beside manifest.jsonl.\n"
        "Audio is mono PCM16 WAV at 16 kHz. Manifest rows contain item_id, source_id, rollout, dialect, voice, text, audio path, sample_rate, and applied augmentations.\n",
        encoding="utf-8")
    print("SYNTH_STATS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
