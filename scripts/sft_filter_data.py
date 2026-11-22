"""Build the B2 SFT data: trajectory rows (B) of the paraphrased utterances only, with the tool list cut to the top-5 apps.

  python scripts/sft_filter_data.py                      # -> work/sft_b2/traj.{train,val}.jsonl + token counts

B2 continues the SFT of B v1 (trained on the organisers' utterances, gold + ASR) on NEW text only, so the train rows are
the "para<n>" rows of work/sft_para/traj.train.jsonl (scripts/plan_data.py --para); every gold / asr row B v1 was
trained on is left out. Each row offers only the tools of the apps in work/retrieval/app_filter.top5.json (RRF all4
top-5 + contacts, plus sys.*), exactly as lifemate.agent.run_sample(keep_apps=...) does at eval (the B + top5 run).
The filter is the one computed on the sample's reference text: the rewrites keep its meaning, and encoding them would
need BGE-M3 (no model runs on the workstation). Every row is kept: when the filter drops an app a trained gold call
uses (82 train rows, 2 val rows), that app's tools are added back for that row, so the model never learns to call a
tool it was not offered. Val = work/sft/traj.val.jsonl with the same filter (val loss).
Per-sample data: everything stays under work/ (git-ignored); prints aggregates only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT  # noqa: E402
from lifemate.sft import encode  # noqa: E402

FILTER = PROJECT / "work/retrieval/app_filter.top5.json"


def app(name: str) -> str:
    return name.split(".")[0]


def cut(row: dict, keep: set[str]) -> tuple[dict, bool]:
    """The row with only the kept apps' tools (+ sys, + any app a trained gold call uses); True when an app was added back."""
    called = {app(c["function"]["name"]) for m in row["messages"] if m.get("train") for c in m.get("tool_calls") or []}
    allowed = keep | {"sys"} | called
    return {**row, "tools": [t for t in row["tools"] if app(t["function"]["name"]) in allowed]}, not called <= keep | {"sys"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="work/sft_para", help="output of scripts/plan_data.py --para")
    ap.add_argument("--out", default="work/sft_b2")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-AWQ", help="only for the token counts (same Qwen3 tokenizer)")
    a = ap.parse_args()
    keep = {i: set(v) for i, v in json.loads(FILTER.read_text(encoding="utf-8"))["keep"]["gold"].items()}
    out = PROJECT / a.out
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    for part, src, want in (("train", PROJECT / a.src / "traj.train.jsonl", lambda v: v.startswith("para")),
                            ("val", PROJECT / "work/sft/traj.val.jsonl", lambda v: v == "gold")):
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        cand = [r for r in rows if want(r["variant"])]
        res = [cut(r, keep[r["id"]]) for r in cand]
        kept, added = [r for r, _ in res], sum(f for _, f in res)
        before = [len(r["tools"]) for r in cand]
        after = [len(r["tools"]) for r in kept]
        n_old = [encode(r, tok, 10 ** 6)["n_tokens"] for r in cand]
        n_new = [encode(r, tok, 10 ** 6)["n_tokens"] for r in kept]
        (out / f"traj.{part}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        print(f"{part}: {len(rows)} rows in, {len(cand)} candidates ({'rewrites' if part == 'train' else 'gold'}), "
              f"gold app added back on {added}, kept {len(kept)} ({len({r['id'] for r in kept})} samples)")
        print(f"   tools per row: mean {st.mean(before):.1f} -> {st.mean(after):.1f} | tokens per row: mean {st.mean(n_old):.0f} -> "
              f"{st.mean(n_new):.0f}, max {max(n_new)} | total {sum(n_old) / 1e6:.2f}M -> {sum(n_new) / 1e6:.2f}M")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
