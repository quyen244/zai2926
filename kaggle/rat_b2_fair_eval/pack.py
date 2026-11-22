"""Package the completed B2 adapter as an explicit Kaggle dataset input."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work/kaggle/b2_train_output_remote/out/traj_b2/adapter"
OUT = ROOT / "work/kaggle_dataset_rat_b2_fair_fold0_adapter"
DATASET_ID = "tquynlnguyn/lifemate-rat-b2-fair-fold0-adapter"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {OUT}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (SOURCE / filename).is_file():
            raise SystemExit(f"missing B2 adapter file: {SOURCE / filename}")

    target = OUT / "rat_adapters" / "traj_b2"
    target.mkdir(parents=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(SOURCE / filename, target / filename)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "lifemate-rat-b2-fair-fold0-adapter",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    size_mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"packed B2 adapter, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
