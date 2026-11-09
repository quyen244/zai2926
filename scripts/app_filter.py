"""Compute the per-sample tool-list filter: keep the top-k installed apps by RRF all4 (+ contacts), for a Kaggle run.

  python scripts/app_filter.py                 # first 100 train samples, k = 5, gold and asr -> work/retrieval/app_filter.top5.json
  python scripts/app_filter.py --limit 0 --k 5 # every train sample

Ranking = scripts/retrieval_sample.py "rrf all4": reciprocal rank fusion of BGE-M3 max, BGE-M3 app doc, BM25 max and
BM25 app doc over that sample's own installed apps (the best app-level setting in reports/retrieval_dense.md,
86.2% sample recall at k = 5 on ASR text over all train). Kept = top-k apps + contacts when installed; sys.* is always
offered by lifemate.agent.run_sample(keep_apps=...). The query is the normalised final turn of the variant: reference
text for "gold" (B0), cached PhoWhisper text for "asr" (B1).
BGE-M3 vectors come from work/retrieval/bge-m3.dense.npz (no model load when every text is cached), so the Kaggle job
only reads this JSON and runs no embedder. Per-sample data: the output stays under work/ (git-ignored). Prints the
recall of the filter on these samples and the tools kept, aggregates only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from retrieval_dense import Encoder  # noqa: E402
from retrieval_sample import rank_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="first N train samples (0 = all), same order as the Kaggle jobs")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    samples = load_samples("train")
    if a.limit:
        samples = samples[: a.limit]
    apps = {x["id"]: x for x in load_catalog("train")["apps"]}
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    enc = Encoder()
    out = {"method": f"rrf all4 top{a.k} + contacts", "k": a.k, "keep": {}}
    for v in ("gold", "asr"):
        keep, ok, n_gold, kept_tools, all_tools = {}, 0, 0, [], []
        for s in samples:
            q = normalize_text(s["audio_text"] if v == "gold" else asr[s["audio"]])
            inst, gold, ranked = rank_sample(s, q, apps, enc)
            k_apps = set([x for x, _ in ranked["rrf all4"]][: a.k]) | ({"contacts"} & set(inst))
            keep[s["id"]] = sorted(k_apps)
            tools = build_prompt(s, "train", s["audio_text"], None)["tools"]
            all_tools.append(len(tools))
            kept_tools.append(sum(t["function"]["name"].split(".")[0] in k_apps | {"sys"} for t in tools))
            if gold:
                n_gold += 1
                ok += gold <= k_apps
        out["keep"][v] = keep
        print(f"{v}: {len(samples)} samples | gold apps all kept on {ok}/{n_gold} samples with a non-sys gold call ({100 * ok / max(1, n_gold):.1f}%)"
              f" | tools offered mean {st.mean(kept_tools):.1f}, median {st.median(kept_tools):.0f} (today {st.mean(all_tools):.1f}, {st.median(all_tools):.0f})")
    dst = PROJECT / "work" / "retrieval" / f"app_filter.top{a.k}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"-> {dst}" + (f" ({enc.new} texts newly encoded)" if enc.new else ""))


if __name__ == "__main__":
    main()
