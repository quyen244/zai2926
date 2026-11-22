from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/tmp/asr")
EXPECTED_CLIPS = 43938
EXPECTED_TRAIN_ROWS = 39172


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def extract(archive: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dst)
    log("extracted", archive.name, "wav files", sum(1 for _ in dst.rglob("*.wav")))
    return dst


def main() -> None:
    inp = Path("/kaggle/input")
    data = next(p.parent for p in inp.rglob("real.jsonl"))
    archive = next(inp.rglob("audio.zip"))
    manifest = archive.parent / "manifest.jsonl"
    qa_path = next(inp.rglob("qa_summary.json"), None)
    if qa_path is None:
        raise SystemExit("missing successful audio_qc job output; refuse to train")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if not qa.get("pass") or qa.get("checked_clips") != EXPECTED_CLIPS:
        raise SystemExit(f"audio QC did not pass for all {EXPECTED_CLIPS} clips: {qa}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_CLIPS:
        raise SystemExit(f"TTS manifest has {len(rows)} rows, expected {EXPECTED_CLIPS}")
    cv = json.loads((data / "cv5.json").read_text(encoding="utf-8"))
    real_rows = [json.loads(line) for line in (data / "real.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_count = sum(r["fold"] != 0 for r in real_rows)
    for row in rows:
        source_id = row["source_id"]
        cv_id = source_id if source_id in cv else source_id.rsplit("_u", 1)[0]
        if cv_id not in cv:
            raise SystemExit(f"source {source_id!r} missing from cv5.json")
        train_count += cv[cv_id] != 0
    if train_count != EXPECTED_TRAIN_ROWS:
        raise SystemExit(f"training row count changed: {train_count}, expected {EXPECTED_TRAIN_ROWS}")
    log("QC gate passed", qa["checked_clips"], "audio clips; training rows", train_count)
    ckpt = next((p.parent for p in inp.rglob("checkpoint-*/trainer_state.json")), None)
    part = 2 if ckpt else 1
    log("data", data, "TTS archive", archive, "part", part, "checkpoint", ckpt)

    real_archive = next(inp.rglob("real.zip"), None)
    if real_archive is not None:
        real = extract(real_archive, TMP / "real")
    else:
        # Kaggle sometimes mounts/extracts dataset files into a nested directory.
        # Resolve audio paths from the manifest instead of assuming real.zip exists.
        first_audio = real_rows[0]["audio"] if real_rows else None
        candidates = [p for p in inp.rglob(first_audio) if p.is_file()] if first_audio else []
        if not candidates:
            raise SystemExit(f"real audio unavailable: no real.zip or mounted {first_audio!r}; inputs={list(inp.iterdir())}")
        real = candidates[0].parents[len(Path(first_audio).parts) - 1]
        log("using mounted real audio directory", real)
    synth = extract(archive, TMP / "synth")
    missing = [r["audio"] for r in rows if not (synth / r["audio"]).is_file()]
    if missing:
        raise SystemExit(f"TTS audio missing after extraction: {len(missing)}; examples={missing[:5]}")

    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1", "accelerate>=1.2",
                    "librosa", "soundfile"], check=True)
    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DATA": str(data), "REAL_DIR": str(real), "SYNTH_DIR": str(synth),
           "SYNTH_MANIFEST": str(manifest), "OUT": str(out), "PART": str(part), "MAX_HOURS": "10.3"}
    if ckpt:
        env["CKPT"] = str(ckpt)
    with (out / "train.log").open("w") as f:
        p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", str(data / "train.py")],
                             env=env, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        while p.poll() is None:
            time.sleep(120)
            tail = (out / "train.log").read_text(errors="ignore").replace("\r", "\n").strip().splitlines()[-1:]
            log(f"{time.time() - t0:.0f}s", tail[0][-200:] if tail else "")
    log("exit code", p.returncode)
    print("--- train.log tail ---\n" + "\n".join((out / "train.log").read_text(errors="ignore").replace("\r", "\n").splitlines()[-30:]))
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
