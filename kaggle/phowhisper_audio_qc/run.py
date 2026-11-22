from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

EXPECTED = 43938
MIN_RMS_DBFS = -38.0
OUT = Path("/kaggle/working/out")


def main() -> None:
    if importlib.util.find_spec("soundfile") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
    import soundfile as sf

    inp = Path("/kaggle/input")
    archives = list(inp.rglob("audio.zip"))
    if len(archives) != 1:
        raise SystemExit(f"expected exactly one TTS audio.zip, found {len(archives)}: {archives}")
    archive = archives[0]
    manifest_path = archive.parent / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest.jsonl missing next to {archive}")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} manifest rows, found {len(rows)}")
    names = [r["audio"] for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("manifest contains duplicate audio paths")

    OUT.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    failures: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        members = {n for n in zf.namelist() if not n.endswith("/")}
        if members != set(names):
            raise SystemExit(f"archive/manifest mismatch: missing={len(set(names)-members)}, extra={len(members-set(names))}")
        for i, row in enumerate(rows, 1):
            with zf.open(row["audio"]) as f:
                info = sf.info(f)
                f.seek(0)
                x, sr = sf.read(f, dtype="float32", always_2d=False)
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
            dbfs = 20 * math.log10(max(rms, 1e-12))
            duration = len(x) / sr if sr else 0.0
            active = float(np.mean(np.abs(x) > 1e-4)) if x.size else 0.0
            rec = {"item_id": row["item_id"], "source_id": row["source_id"], "dialect": row["dialect"],
                   "augmentations": ",".join(row.get("augmentations", [])), "sample_rate": sr,
                   "channels": info.channels, "subtype": info.subtype, "duration_s": round(duration, 3),
                   "rms_dbfs": round(dbfs, 2), "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 2),
                   "active_fraction": round(active, 4)}
            stats.append(rec)
            reason = []
            if sr != 16000 or info.channels != 1 or info.subtype != "PCM_16":
                reason.append("format must be mono 16 kHz PCM16")
            if not np.isfinite(x).all():
                reason.append("non-finite samples")
            if duration < 0.25 or duration > 30:
                reason.append(f"duration out of range: {duration:.3f}s")
            if dbfs < MIN_RMS_DBFS:
                reason.append(f"quiet: RMS {dbfs:.2f} dBFS")
            if peak <= 1e-4 or active < 0.01:
                reason.append("silence/near-silence")
            if peak >= 0.999:
                reason.append(f"clipped peak: {peak:.4f}")
            if reason:
                failures.append({"item_id": row["item_id"], "audio": row["audio"], "reasons": reason})
            if i % 5000 == 0:
                print(f"checked {i}/{len(rows)}", flush=True)

        candidates = []
        for i, row in enumerate(rows):
            rec = stats[i]
            if rec["rms_dbfs"] >= MIN_RMS_DBFS:
                candidates.append((i, row, rec))
        chosen: dict[str, tuple[int, dict, dict]] = {}
        for label, predicate in (
            ("target_south_reverb", lambda r: r["item_id"] == "train_0127_r1_south" and "reverb" in r["augmentations"]),
            ("north_clean", lambda r: r["dialect"] == "north" and not r["augmentations"]),
            ("central_reverb", lambda r: r["dialect"] == "central" and "reverb" in r["augmentations"]),
            ("south_noise", lambda r: r["dialect"] == "south" and "noise_snr" in r["augmentations"]),
        ):
            found = next((x for x in candidates if predicate(x[2])), None)
            if found:
                chosen[label] = found
        rev = [x for x in candidates if "reverb" in x[2]["augmentations"]]
        if rev:
            chosen["reverb_quietest_pass"] = min(rev, key=lambda x: x[2]["rms_dbfs"])

        preview_dir = OUT / "previews"
        preview_dir.mkdir(exist_ok=True)
        previews = []
        for label, (i, row, rec) in chosen.items():
            dst = preview_dir / f"{label}.wav"
            dst.write_bytes(zf.read(row["audio"]))
            previews.append({"label": label, "path": dst.name, "item_id": row["item_id"],
                             "text": row["text"], "rms_dbfs": rec["rms_dbfs"], "augmentations": rec["augmentations"]})

    rms = np.array([r["rms_dbfs"] for r in stats], dtype=np.float64)
    def group_percentiles(group: list[dict]) -> dict:
        values = [r["rms_dbfs"] for r in group]
        return {"count": len(values), **{str(q): round(float(np.percentile(values, q)), 2) for q in (0, 10, 50, 90, 100)}} if values else {"count": 0}

    reverb_stats = [r for r in stats if "reverb" in r["augmentations"]]
    clean_stats = [r for r in stats if "reverb" not in r["augmentations"]]
    summary = {"pass": not failures, "expected_clips": EXPECTED, "checked_clips": len(stats),
               "min_rms_dbfs_gate": MIN_RMS_DBFS, "failed_clips": len(failures),
               "rms_dbfs_percentiles": {str(q): round(float(np.percentile(rms, q)), 2) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
               "reverb_rms_dbfs": group_percentiles(reverb_stats), "non_reverb_rms_dbfs": group_percentiles(clean_stats),
               "previews": previews,
               "failures_sample": failures[:100]}
    (OUT / "qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "audio_qc.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f"AUDIO QC FAILED: {len(failures)} clips failed. Training must not start.")
    print("AUDIO QC PASSED: every clip is readable, audible-level, mono 16 kHz PCM16, and unclipped.", flush=True)


if __name__ == "__main__":
    main()
