"""Per sample: rank that sample's installed apps by BM25, BGE-M3 and their fusion, and check whether top-k keeps the gold apps.

  python scripts/retrieval_sample.py train_0018                 # one sample, ASR text, k = 4 and 5
  python scripts/retrieval_sample.py train_0018 train_0007 --text gold
  python scripts/retrieval_sample.py --all                      # every train sample: recall of top-4 / top-5 (screen only)

Candidates = the apps whose tools the prompt gets for that sample (installed apps, sys excluded), so every sample is
ranked over its own app list. Context per app = the app doc of scripts/retrieval_dense.py (id + name_vi + description).
Rankings shown:
  bm25 app    BM25 of the query against each installed app's doc (IDF over that sample's apps)
  dense app   BGE-M3 cosine of the query and each app doc
  rrf all4    reciprocal rank fusion of dense max, dense app, BM25 max, BM25 app (max = best tool of the app)
Two keep rules: "top-k" alone, and "top-k + contacts" (contacts always kept, the rule of reports/retrieval_dense.md).
BGE-M3 vectors come from work/retrieval/bge-m3.dense.npz; a text not cached there is encoded on CPU (slow).
Prints ids and queries to the screen only (public repo): nothing is written to reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from retrieval_dense import Encoder, app_doc, rrf, tool_doc  # noqa: E402
from retrieval_recall import bm25, tokens  # noqa: E402

RANKERS = ["bm25 app", "dense app", "rrf all4"]


def rank_sample(s: dict, query: str, apps: dict, enc: Encoder) -> tuple[list[str], set[str], dict[str, list[tuple[str, float]]]]:
    tools = [t for t in build_prompt(s, "train", s["audio_text"], None)["tools"] if not t["function"]["name"].startswith("sys.")]
    names = [t["function"]["name"] for t in tools]
    inst = sorted({n.split(".")[0] for n in names})
    gold = {c["name"].split(".")[0] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}
    qv = enc.encode([query])[0]
    dtool = list(enc.encode([tool_doc(t) for t in tools]) @ qv)
    dapp = dict(zip(inst, (float(x) for x in enc.encode([app_doc(apps[x]) for x in inst]) @ qv)))
    qt = tokens(query, False)
    btool = bm25(qt, [tokens(tool_doc(t), False) for t in tools])
    bapp = dict(zip(inst, bm25(qt, [tokens(app_doc(apps[x]), False) for x in inst])))

    def by_max(sc: list[float]) -> list[str]:
        m: dict[str, float] = {}
        for n, x in zip(names, sc):
            a = n.split(".")[0]
            m[a] = max(m.get(a, float("-inf")), float(x))
        return sorted(inst, key=lambda a: -m[a])

    order = lambda d: sorted(inst, key=lambda a: -d[a])  # noqa: E731
    fused = rrf(by_max(dtool), order(dapp), by_max(btool), order(bapp))
    out = {"bm25 app": [(a, bapp[a]) for a in order(bapp)],
           "dense app": [(a, dapp[a]) for a in order(dapp)],
           "rrf all4": [(a, 0.0) for a in fused]}
    return inst, gold, out


def kept(ranked: list[str], gold: set[str], k: int, contacts: bool) -> bool:
    keep = set(ranked[:k]) | ({"contacts"} if contacts else set())
    return gold <= keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--text", choices=["asr", "gold"], default="asr")
    ap.add_argument("--k", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--all", action="store_true", help="every train sample with a non-sys gold tool: recall table")
    a = ap.parse_args()
    if not a.ids and not a.all:
        ap.error("give sample ids or --all")

    samples = {s["id"]: s for s in load_samples("train")}
    apps = {x["id"]: x for x in load_catalog("train")["apps"]}
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    enc = Encoder()
    query = lambda s: normalize_text(s["audio_text"] if a.text == "gold" else asr[s["audio"]])  # noqa: E731

    for sid in a.ids:
        if sid not in samples:
            print(f"{sid}: not a train sample id")
            continue
        s = samples[sid]
        q = query(s)
        inst, gold, out = rank_sample(s, q, apps, enc)
        print(f"\n== {sid} ({a.text}) | {len(inst)} installed apps | gold apps: {', '.join(sorted(gold)) or '(none, only sys)'}")
        print(f"   query: {q}")
        for r in RANKERS:
            row = "  ".join(f"{i}.{'*' if x in gold else ''}{x}" + (f" {sc:.2f}" if r != "rrf all4" else "") for i, (x, sc) in enumerate(out[r], 1))
            ranked = [x for x, _ in out[r]]
            verdict = "  ".join(f"top{k} {'ok' if kept(ranked, gold, k, False) else 'MISS'} (+contacts {'ok' if kept(ranked, gold, k, True) else 'MISS'})" for k in a.k)
            print(f"   {r:9s} | {verdict}\n             {row}")

    if a.all:
        rows = []
        for s in samples.values():
            if not any(not c["name"].startswith("sys.") for c in s["label"]["tool_calls"]):
                continue  # nothing to keep: gold empty or only sys.ask_missing_info
            inst, gold, out = rank_sample(s, query(s), apps, enc)
            rows.append((len(inst), {r: [x for x, _ in out[r]] for r in RANKERS}, gold))
        n = len(rows)
        print(f"\n== all train, {a.text} text: {n} samples with a non-sys gold tool; installed apps per sample {min(r[0] for r in rows)} to {max(r[0] for r in rows)}")
        print("   sample recall % (gold apps all kept)")
        print("   ranker    | " + " | ".join(f"top{k} | top{k}+contacts" for k in a.k))
        for r in RANKERS:
            cells = []
            for k in a.k:
                cells += [f"{100 * sum(kept(x[1][r], x[2], k, False) for x in rows) / n:5.1f}", f"{100 * sum(kept(x[1][r], x[2], k, True) for x in rows) / n:5.1f}"]
            print(f"   {r:9s} | " + " | ".join(cells))
        small = {k: sum(1 for x in rows if x[0] <= k) for k in a.k}
        print("   samples with <= k installed apps (top-k keeps everything): " + ", ".join(f"k={k}: {v}" for k, v in small.items()))
    if enc.new:
        print(f"({enc.new} texts newly encoded and cached)")


if __name__ == "__main__":
    main()
