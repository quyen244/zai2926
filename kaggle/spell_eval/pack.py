"""Package PhoWhisper-FT text/reference pairs for the Kaggle correction benchmark."""
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/kaggle_dataset_spell_eval"
DATA = ROOT / "data/train+public_test/train/train.jsonl"
ASR = ROOT / "work/asr/phowhisper_ft/train.jsonl"
CV = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
samples = {s["id"]: s for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip() for s in [json.loads(line)]}
asr = {r["audio"]: r["hyp"] for line in ASR.read_text(encoding="utf-8").splitlines() if line.strip() for r in [json.loads(line)] if r.get("audio")}
rows = [{"id": sid, "split": "eval" if CV[sid] == 0 else "train", "ref": s["audio_text"], "hyp": asr[s["audio"]]}
        for sid, s in samples.items() if s["audio"] in asr]
if len(rows) != 1545 or sum(r["split"] == "eval" for r in rows) != 309:
    raise SystemExit(f"expected 1545 final-turn ASR pairs, 309 held-out fold 0; got {len(rows)} / {sum(r['split']=='eval' for r in rows)}")
OUT.mkdir(parents=True, exist_ok=True)
with tarfile.open(OUT / "examples.tar.gz", "w:gz") as tar:
    p = OUT / "examples.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tar.add(p, arcname="examples.jsonl")
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-spell-correction-input", "id": "quyenuit24/lifemate-spell-correction-input", "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
print(f"{len(rows)} pairs, held-out {sum(r['split']=='eval' for r in rows)}, {OUT / 'examples.tar.gz'}")
