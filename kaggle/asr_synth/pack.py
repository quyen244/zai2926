from __future__ import annotations

import json
import random
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "kaggle_asr_synth_input"
ROLLOUTS = ROOT / "work" / "synth" / "asr_rollouts_v2.jsonl"
K = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="package every source with three valid paraphrases")
    ap.add_argument("--complete", action="store_true", help="package every source that already has three paraphrases (partial run)")
    ap.add_argument("--leftovers", action="store_true", help="also voice sources with < 3 paraphrases: their original text + whatever paraphrases they have (no new generation)")
    ap.add_argument("--skip-done", action="store_true", help="leave out sources already voiced in batch 1 (asr_synth_full manifest)")
    args = ap.parse_args()
    if not ROLLOUTS.is_file():
        raise SystemExit(f"Missing generated rollout cache: {ROLLOUTS}")
    latest = {}
    for line in ROLLOUTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    complete = [r for r in latest.values() if len(r.get("rollouts") or []) >= K]
    short = []
    if args.leftovers:
        short = [{**r, "rollouts": [r["text"]] + list(r.get("rollouts") or [])} for r in latest.values() if len(r.get("rollouts") or []) < K]
        print(f"leftover sources voiced from their original text: {len(short)}")
    complete.sort(key=lambda r: r["id"])
    if args.all and len(complete) + (len(short) if args.leftovers else 0) != 5036:
        raise SystemExit(f"Full job requires 5,036 sources (including leftovers when requested), currently have {len(complete) + (len(short) if args.leftovers else 0)}")
    if args.skip_done:
        man = ROOT / "work" / "kaggle" / "asr_synth_full" / "out" / "manifest.jsonl"
        done = {json.loads(l)["source_id"] for l in man.read_text(encoding="utf-8").splitlines() if l.strip()}
        complete = [r for r in complete + (short if args.leftovers else []) if r["id"] not in done]
        complete.sort(key=lambda r: r["id"])
        print(f"skipping {len(done)} sources voiced in batch 1")
    elif args.leftovers:
        complete = complete + short
    if not complete:
        raise SystemExit("No complete source rows available for the smoke run")
    selected = complete if args.all or args.complete else random.Random(42).sample(complete, min(96, len(complete)))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = [{"id": r["id"], "source_text": r["text"], "rollouts": r["rollouts"][:3]} for r in selected]
    (OUT / "rollouts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    shutil.copyfile(ROOT / "kaggle" / "asr_synth" / "worker.py", OUT / "worker.py")
    shutil.copyfile(ROOT / "work" / "listening" / "listening.csv", OUT / "listening.csv")
    meta = {"title": "LifeMate ASR synthetic speech smoke input", "id": "tquynlnguyn/lifemate-asr-synth-smoke-input",
            "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    title = "LifeMate ASR synthetic speech input"
    meta = {"title": title, "id": "tquynlnguyn/lifemate-asr-synth-input", "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} sources, {sum(len(r['rollouts']) for r in rows)*3} expected clips")


if __name__ == "__main__":
    main()
