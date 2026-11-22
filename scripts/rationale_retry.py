"""Drop the cached rationale replies that fail the code check, so the next `rationale_data.py --run` asks again.

  python scripts/rationale_retry.py --text gold   # moves failing rows to work/sft/rationale.<text>.rejected.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rationale_data as R  # noqa: E402
from lifemate.data import load_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", choices=["gold", "asr"], default="gold")
    a = ap.parse_args()
    S = {s["id"]: s for s in load_samples("train")}
    done = R.load(a.text)
    bad = {i for i, r in done.items()
           if any(x["bad"] for x in R.check(S[i], R.steps_for(S[i], a.text), r.get("content", ""), a.text))}
    f = R.OUT[a.text]
    with f.with_suffix(".rejected.jsonl").open("a", encoding="utf-8") as g:
        for i in sorted(bad):
            g.write(json.dumps(done[i], ensure_ascii=False) + "\n")
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for i, r in done.items() if i not in bad), encoding="utf-8")
    print(f"{a.text}: {len(done)} cached, {len(bad)} failing moved out, {len(done) - len(bad)} kept")


if __name__ == "__main__":
    main()
