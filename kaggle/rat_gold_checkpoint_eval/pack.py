"""Package the paired fold-0 inputs and top-7 embedder for the gold RAT checkpoint evaluation."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "work/kaggle_dataset_rat_gold_checkpoint_eval"
DATA = ROOT / "data/train+public_test/train"
ASR_TOP = ROOT / "work/kaggle/asr_p2_decode_sweep_tquyen/out/baseline.jsonl"
ASR_TRAIN = ROOT / "work/kaggle/asr_p2_decode_train_pairs/out/train_p2.jsonl"
E5 = ROOT / "work/kaggle/embed_train_q/out/app_embedder_e5"
GOLD_ADAPTER = ROOT / "work/kaggle/rat_gold_para_top7_train_output/out/traj_rat_gold_para_top7/adapter"
DATASET_ID = "tquynlnguyn/lifemate-rat-gold-checkpoint-eval-input-v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {OUT}")
    cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
    test = read_jsonl(ROOT / "work/sft/plan.test.jsonl")
    ids = {row["id"] for row in test}
    expected = {sample_id for sample_id, fold in cv.items() if fold == 0}
    if ids != expected or len(ids) != 309:
        raise SystemExit(f"fold-0 test ids do not match: {len(ids)} ids")

    from lifemate.data import load_samples
    samples = {row["id"]: row for row in load_samples("train") if row["id"] in ids}
    if set(samples) != ids:
        raise SystemExit("missing fold-0 train rows")
    needed_audio = set()
    for sample in samples.values():
        if sample.get("audio"):
            needed_audio.add(sample["audio"])
        needed_audio.update(turn["audio"] for turn in sample.get("history", [])
                            if turn.get("role") == "user" and turn.get("audio"))

    p2 = {}
    for path in (ASR_TRAIN, ASR_TOP):
        for row in read_jsonl(path):
            audio = row["audio"]
            if audio in p2:
                raise SystemExit(f"duplicate P2 audio row: {audio}")
            p2[audio] = {"audio": audio, "hyp": row["hyp"], "avg_logprob": row.get("avg_logprob")}
    if needed_audio - p2.keys():
        raise SystemExit(f"missing P2 transcript for {len(needed_audio - p2.keys())} fold-0 turns")
    p2_rows = [p2[a] for a in sorted(needed_audio)]

    repo = OUT / "repo"
    (repo / "lifemate").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "lifemate", repo / "lifemate", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dest = repo / "data/train+public_test/train"
    dest.mkdir(parents=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(samples[i], ensure_ascii=False) + "\n" for i in sorted(ids)), encoding="utf-8")
    shutil.copy2(DATA / "train_tools_catalog.json", dest / "train_tools_catalog.json")
    shutil.copytree(DATA / "zaic_tools", dest / "zaic_tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    envdst = dest / "train_environment"
    envdst.mkdir()
    for sample_id in sorted(ids):
        shutil.copytree(DATA / "train_environment" / sample_id, envdst / sample_id,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    (repo / "splits").mkdir()
    shutil.copy2(ROOT / "splits/cv5.json", repo / "splits/cv5.json")
    (repo / "work/sft").mkdir(parents=True)
    shutil.copy2(ROOT / "work/sft/plan.test.jsonl", repo / "work/sft/plan.test.jsonl")
    asr_dest = repo / "work/asr/phowhisper_p2"
    asr_dest.mkdir(parents=True)
    (asr_dest / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in p2_rows), encoding="utf-8")

    shutil.copytree(E5, OUT / "app_embedder_e5", ignore=shutil.ignore_patterns("train.log", "best_config.json"))
    if not (GOLD_ADAPTER / "adapter_config.json").is_file() or not (GOLD_ADAPTER / "adapter_model.safetensors").is_file():
        raise SystemExit(f"missing gold checkpoint adapter files: {GOLD_ADAPTER}")
    target = OUT / "rat_adapters" / "traj_rat_gold"
    target.mkdir(parents=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(GOLD_ADAPTER / filename, target / filename)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "lifemate-rat-gold-checkpoint-eval-input-v2",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")

    for needed in ("lifemate/agent.py", "data/train+public_test/train/train.jsonl",
                   "work/sft/plan.test.jsonl", "work/asr/phowhisper_p2/train.jsonl"):
        if not (repo / needed).is_file():
            raise SystemExit(f"missing from repo folder: {needed}")
    if not (OUT / "app_embedder_e5/config.json").is_file() or not (OUT / "app_embedder_e5/model.safetensors").is_file():
        raise SystemExit("top-7 embedder checkpoint is incomplete")
    size_mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"packed {len(ids)} fold-0 samples, {len(needed_audio)} P2 turns, {len(p2_rows)} P2 rows, new RAT adapter only, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
