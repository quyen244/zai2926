"""Stage 3: run a baseline (B0/B1/B2) over a split and write a submission.jsonl in the organiser's exact format
({"id", "tool_calls", "state_hash"}, verified against data/train+public_test/public_test/sample_submission.jsonl).

  python scripts/run_baseline.py --split train --text gold --limit 20                       # B0 smoke test
  python scripts/run_baseline.py --split public_test --text asr --limit 20                  # B1 smoke test
  python scripts/run_baseline.py --split public_test --text asr                             # B1 full run
  python scripts/run_baseline.py --split public_test --text asr --validate-first            # B2

--text gold only works on train (needs sample["audio_text"]); public_test always needs --text asr.
Resumable: already-written ids in --out are skipped. Writes to work/submissions/ (git-ignored) by default.
When the split has gold labels (train), also scores the run locally with lifemate.scoring after it finishes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.agent import run_sample  # noqa: E402
from lifemate.data import PROJECT, fuzzy_params, load_samples  # noqa: E402
from lifemate.gold import build_gold, validate_submission  # noqa: E402
from lifemate.llm import LLM, load_env  # noqa: E402
from lifemate.scoring import score_split  # noqa: E402

OUT_DIR = PROJECT / "work" / "submissions"


def load_asr(stem: str) -> dict[str, str]:
    path = PROJECT / "work" / "asr" / f"{stem}.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    return {json.loads(line)["audio"]: json.loads(line)["hyp"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--text", choices=["gold", "asr"], default="asr")
    ap.add_argument("--asr", help="work/asr/<stem>.jsonl; default phowhisper/<split>")
    ap.add_argument("--limit", type=int, help="only the first N unrun samples")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step-cap", type=int, default=6)
    ap.add_argument("--validate-first", action="store_true", help="B2: free env.validate() before every env.call()")
    ap.add_argument("--out", type=Path, help="default work/submissions/<split>.<text>.<model>.jsonl")
    a = ap.parse_args()

    if a.text == "gold" and a.split != "train":
        raise SystemExit("--text gold only works on --split train (no other split has a gold transcript)")

    load_env()
    model = os.environ.get("LLM_MODEL", "gpt-6-luna")
    llm = LLM(model=model)

    samples = load_samples(a.split)
    asr = {}
    if a.text == "asr":
        asr = load_asr(a.asr or f"phowhisper/{a.split}")
    text_of = (lambda s: s["audio_text"]) if a.text == "gold" else (lambda s: asr[s["audio"]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = a.out or OUT_DIR / f"{a.split}.{a.text}.{model}.jsonl"
    done: dict[str, dict] = {}
    if out_path.is_file():
        done = {json.loads(line)["id"]: json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in samples if s["id"] not in done][: a.limit]
    print(f"{len(done)} already done, {len(todo)} to run on {a.split} ({a.text}) with {model}, step_cap={a.step_cap}, validate_first={a.validate_first}")

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="\n") as fout, ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(run_sample, s, a.split, text_of(s), llm, asr or None, a.step_cap, a.validate_first): s["id"] for s in todo}
        n = 0
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"FAILED {sid}: {type(e).__name__}: {e}")
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    u = llm.usage
    print(f"requests={u.requests} retries={u.retries} prompt_tokens={u.prompt_tokens} completion_tokens={u.completion_tokens} sec={time.time() - t0:.0f}")

    problems = validate_submission(out_path, a.split)
    errors = [p for p in problems if not p.startswith("warn:")]
    warnings = [p for p in problems if p.startswith("warn:")]
    print(f"validate_submission: {len(errors)} error(s), {len(warnings)} warning(s)" + (f" (first: {errors[0]})" if errors else ""))

    if a.split == "train":
        gold = build_gold("train")
        preds = {json.loads(line)["id"]: json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        preds = {k: v for k, v in preds.items() if k in gold}
        if preds:
            final, _ = score_split(preds, {k: gold[k] for k in preds}, fuzzy_params("train"))
            print(f"local score on {len(preds)} scored train samples: {final:.2f}")

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
