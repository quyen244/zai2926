from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_phowhisper_smoke_input"
TRAIN = ROOT / "data" / "train+public_test" / "train"
SEED = 42


def main() -> None:
    rows = [json.loads(line) for line in (TRAIN / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    utterances: list[dict] = []
    for sample in rows:
        if sample.get("audio") and sample.get("audio_text"):
            utterances.append({"id": sample["id"], "audio": sample["audio"], "text": sample["audio_text"]})
        for i, turn in enumerate(sample.get("history", [])):
            if turn.get("audio") and turn.get("audio_text"):
                utterances.append({"id": f"{sample['id']}_u{i+1}", "audio": turn["audio"], "text": turn["audio_text"]})
    random.Random(SEED).shuffle(utterances)
    chosen = utterances[:128]
    if len(chosen) < 128:
        raise SystemExit(f"Need 128 labelled train utterances, found {len(chosen)}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    for row in chosen:
        src = TRAIN / row["audio"]
        dst = OUT / "audio" / (row["id"] + ".wav")
        shutil.copyfile(src, dst)
        row["audio"] = "audio/" + dst.name
    manifest = OUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    shutil.copyfile(ROOT / "kaggle" / "phowhisper_smoke" / "train.py", OUT / "train.py")
    meta = {"title": "PhoWhisper smoke train audio", "id": "tquynlnguyn/phowhisper-smoke-audio",
            "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Prepared {len(chosen)} train utterances (112 train / 16 held-out), {sum(p.stat().st_size for p in (OUT / 'audio').glob('*.wav'))/1e6:.1f} MB audio")


if __name__ == "__main__":
    main()
