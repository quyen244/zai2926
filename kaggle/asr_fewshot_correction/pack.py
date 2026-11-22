"""Package existing fold-0 P2 decode and real-clip references for private Kaggle WER evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/kaggle_dataset_asr_fewshot"
REAL = ROOT / "work/kaggle_phowhisper_input/real.jsonl"
BASELINE = ROOT / "work/kaggle/asr_p2_decode_sweep_tquyen/out/baseline.jsonl"


def main() -> None:
    real = [json.loads(x) for x in REAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    baseline = [json.loads(x) for x in BASELINE.read_text(encoding="utf-8").splitlines() if x.strip()]
    fold0 = {r["audio"] for r in real if r["fold"] == 0}
    if len(real) != 5036 or len(fold0) != 1009 or {r["audio"] for r in baseline} != fold0:
        raise SystemExit("fold-0 P2 baseline does not match real clip manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL, OUT / "real.jsonl")
    shutil.copy2(BASELINE, OUT / "baseline.jsonl")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "lifemate-asr-fewshot-input",
        "id": "tquynlnguyn/lifemate-asr-fewshot-input", "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"packed {len(real)} manifest rows; {len(baseline)} P2 fold-0 transcripts")


if __name__ == "__main__":
    main()
