"""Build the dataset for the PhoWhisper LoRA job (kaggle/phowhisper_train/run.py).

work/kaggle_phowhisper_input/ (private dataset tquynlnguyn/lifemate-phowhisper-data):
  real.zip              every train user-turn recording (audio/<sample>/<clip>.wav)
  real.jsonl            {"id","audio","text","fold"}: gold transcript (work/synth/asr_rollouts_v2.jsonl "text") + CV fold
  synth1.zip            TTS batch 1 audio (work/kaggle/asr_synth_full/out/audio.zip, copied as is)
  synth1.jsonl          its manifest, each row with "fold" of its source sample
  train.py, cv5.json    trainer + folds (batch 2 manifest comes from kernel output and is filtered in train.py)
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_phowhisper_input"
CV = json.loads((ROOT / "splits" / "cv5.json").read_text(encoding="utf-8"))


def fold(clip_id: str) -> int:
    return CV[clip_id.rsplit("_u", 1)[0]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for line in (ROOT / "work/synth/asr_rollouts_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = {"id": r["id"], "audio": r["audio"], "text": r["text"], "fold": fold(r["id"])}
    real = sorted(rows.values(), key=lambda r: r["id"])
    src = ROOT / "data/train+public_test/train"
    missing = [r["audio"] for r in real if not (src / r["audio"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} real clips missing, e.g. {missing[:3]}")
    (OUT / "real.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in real), encoding="utf-8")
    z = OUT / "real.zip"
    if not z.exists():
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for a in sorted({r["audio"] for r in real}):
                zf.write(src / a, a)
    syn = []
    for line in (ROOT / "work/kaggle/asr_synth_full/out/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            syn.append({**r, "fold": fold(r["source_id"])})
    (OUT / "synth1.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in syn), encoding="utf-8")
    if not (OUT / "synth1.zip").exists():
        shutil.copyfile(ROOT / "work/kaggle/asr_synth_full/out/audio.zip", OUT / "synth1.zip")
    shutil.copyfile(Path(__file__).with_name("train.py"), OUT / "train.py")
    shutil.copyfile(ROOT / "splits/cv5.json", OUT / "cv5.json")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-phowhisper-data", "id": "tquynlnguyn/lifemate-phowhisper-data",
                                                           "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    n_real = sum(r["fold"] != 0 for r in real)
    n_syn = sum(r["fold"] != 0 for r in syn)
    print(f"real clips {len(real)} (train folds 1-4: {n_real}, fold 0 val: {len(real) - n_real})")
    print(f"synth batch 1 clips {len(syn)} (folds 1-4: {n_syn})")


if __name__ == "__main__":
    main()
