"""Offline check, dense version: does BGE-M3 retrieval keep the needed tools better than BM25?

  python scripts/retrieval_dense.py --limit 20        # smoke test on the first 20 train samples
  python scripts/retrieval_dense.py                   # all train samples -> reports/retrieval_dense.md (aggregates only)
  python scripts/retrieval_dense.py --misses 10       # also print 10 misses of the best app-level config (screen only)

Same candidates and metrics as scripts/retrieval_recall.py (BM25), so the numbers compare one to one:
candidates = the tools today's prompt gets (installed apps only, sys.* excluded), gold = distinct non-sys gold tools.
Context ("name + description") is the same for both retrievers:
  tool doc : tool name split on . and _ + tool description
  app doc  : app id split on _ + app name_vi + app description
Retrievers
  dense    BAAI/bge-m3 dense vector (CLS, L2-normalised, max 512 tokens), cosine query . doc
  bm25     k1=1.5, b=0.75, IDF over that sample's candidates (as in retrieval_recall.py), accents kept
Levels
  tool     sample recall@k = share of samples whose gold tools are ALL in the top-k tools
  app      apps ranked three ways: max or sum of their tools' scores, or the app doc itself ("app");
           keep the top-k apps with all their tools, plus contacts when installed (same rule as retrieval_recall.py)
  rrf      reciprocal rank fusion (k=60) of the dense and BM25 rankings of the same kind; "all4" fuses dense max,
           dense app, BM25 max and BM25 app
Query = normalize_text(final turn), gold transcript or cached PhoWhisper ASR. No history.
Runs on CUDA when available (Kaggle job kaggle/dense_retrieval/), else CPU; embeddings are cached in work/retrieval/ so a rerun only encodes new texts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
from retrieval_recall import bm25, tokens  # noqa: E402

MODEL = "BAAI/bge-m3"
TOOL_K = [3, 5, 8, 10, 15, 20]
APP_K = [1, 2, 3, 4, 5]
AGGS = ["max", "sum", "app"]
RETRS = ["dense", "bm25", "rrf"]
RRF_K = 60
CACHE = PROJECT / "work" / "retrieval"
REPORT = PROJECT / "reports" / "retrieval_dense.md"


class Encoder:
    """bge-m3 dense embeddings with an on-disk cache keyed by the text's sha1."""

    def __init__(self, batch: int = 16):
        self.path = CACHE / "bge-m3.dense.npz"
        self.vec: dict[str, np.ndarray] = {}
        if self.path.is_file():
            z = np.load(self.path)
            self.vec = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.batch, self.model, self.tok, self.new = batch, None, None, 0

    @staticmethod
    def key(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        """One model copy per visible GPU (data sharding, like the Kaggle LLM jobs); CPU when there is none."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
        self.models = [AutoModel.from_pretrained(MODEL).eval().to(d) for d in self.devs]
        print(f"  {MODEL} on {', '.join(self.devs)}", flush=True)

    def _run_shard(self, k: int, texts: list[str]) -> list[np.ndarray]:
        import torch
        out, t0 = [], time.time()
        for i in range(0, len(texts), self.batch):
            enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.devs[k])
            with torch.no_grad():
                cls = self.models[k](**enc).last_hidden_state[:, 0]
            out += list(torch.nn.functional.normalize(cls.float(), dim=-1).cpu().numpy().astype(np.float32))
            if (i // self.batch) % 20 == 0:
                print(f"  [{self.devs[k]}] {i + self.batch}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = sorted({t for t in texts if self.key(t) not in self.vec}, key=len)
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            if self.model is None:
                self._load()
                self.model = True
            n = len(self.devs)
            shards = [todo[k::n] for k in range(n)]  # interleaved so both shards get the same length mix
            with ThreadPoolExecutor(n) as pool:
                vecs = list(pool.map(self._run_shard, range(n), shards))
            for sh, vs in zip(shards, vecs):
                for t, v in zip(sh, vs):
                    self.vec[self.key(t)] = v
            self.new += len(todo)
            self.save()
        return np.stack([self.vec[self.key(t)] for t in texts])

    def save(self) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        keys = list(self.vec)
        np.savez(self.path, keys=np.array(keys), vecs=np.stack([self.vec[k] for k in keys]))


def rrf(*lists: list[str]) -> list[str]:
    """Reciprocal rank fusion: score(x) = sum over lists of 1 / (RRF_K + rank), rank from 1; ties keep the first list's order."""
    sc: dict[str, float] = {}
    for lst in lists:
        for r, x in enumerate(lst, 1):
            sc[x] = sc.get(x, 0.0) + 1 / (RRF_K + r)
    first = {x: j for j, x in enumerate(lists[0])}
    return sorted(sc, key=lambda x: (-sc[x], first.get(x, len(first))))


def tool_doc(t: dict) -> str:
    f = t["function"]
    return f"{f['name'].replace('.', ' ').replace('_', ' ')}: {f.get('description', '')}"


def app_doc(a: dict) -> str:
    return f"{a['id'].replace('_', ' ')} {a.get('name_vi', '')}: {a.get('description', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N train samples (smoke test)")
    ap.add_argument("--misses", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    samples = load_samples("train")
    if a.limit:
        samples = samples[: a.limit]
    apps = {x["id"]: x for x in load_catalog("train")["apps"]}
    asr = {json.loads(l)["audio"]: json.loads(l)["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    prep = []
    for s in samples:
        tools = [t for t in build_prompt(s, "train", s["audio_text"], None)["tools"] if not t["function"]["name"].startswith("sys.")]
        gold = {c["name"] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}
        if gold:
            prep.append((s, tools, gold))
    queries = {src: [normalize_text(s["audio_text"] if src == "gold" else asr[s["audio"]]) for s, _, _ in prep] for src in ("gold", "asr")}

    enc = Encoder(a.batch)
    all_tools = sorted({t["function"]["name"]: t for _, ts, _ in prep for t in ts}.items())
    t0 = time.time()
    tvec = dict(zip([n for n, _ in all_tools], enc.encode([tool_doc(t) for _, t in all_tools])))
    avec = dict(zip(list(apps), enc.encode([app_doc(x) for x in apps.values()])))
    qvec = {src: enc.encode(q) for src, q in queries.items()}
    print(f"embeddings ready in {time.time() - t0:.0f}s ({enc.new} newly encoded)")

    # per sample: tool order and app ranking per aggregation, for each retriever; "rrf" fuses dense and bm25
    rank: dict[tuple[str, str], list] = {}  # (retr, src) -> per sample {"tool": [...], "max": [...], "sum": [...], "app": [...]}
    for src in ("gold", "asr"):
        for retr in ("dense", "bm25"):
            per = []
            for i, (s, tools, gold) in enumerate(prep):
                names = [t["function"]["name"] for t in tools]
                inst = sorted({nm.split(".")[0] for nm in names})
                if retr == "dense":
                    q = qvec[src][i]
                    sc = [float(q @ tvec[nm]) for nm in names]
                    app_sc = {x: float(q @ avec[x]) for x in inst}
                else:
                    qt = tokens(queries[src][i], False)
                    sc = bm25(qt, [tokens(tool_doc(t), False) for t in tools])
                    app_sc = dict(zip(inst, bm25(qt, [tokens(app_doc(apps[x]), False) for x in inst])))
                by_app: dict[str, list[float]] = {}
                for nm, x in zip(names, sc):
                    by_app.setdefault(nm.split(".")[0], []).append(x)
                r = {"tool": [names[j] for j in sorted(range(len(names)), key=lambda j: -sc[j])], "app": sorted(inst, key=lambda x: -app_sc[x])}
                for g, f in (("max", max), ("sum", sum)):
                    r[g] = sorted(inst, key=lambda x, f=f: -f(by_app[x]))
                per.append(r)
            rank[(retr, src)] = per
        rank[("rrf", src)] = [{g: rrf(d[g], b[g]) for g in ["tool"] + AGGS} for d, b in zip(rank[("dense", src)], rank[("bm25", src)])]
        # all four strong app lists at once: dense max, dense app, bm25 max, bm25 app
        for d, b, f in zip(rank[("dense", src)], rank[("bm25", src)], rank[("rrf", src)]):
            f["all4"] = rrf(d["max"], d["app"], b["max"], b["app"])

    results, misses = [], {}
    for retr in RETRS:
        for src in ("gold", "asr"):
            name = f"{retr}, {src}"
            aggs = AGGS + (["all4"] if retr == "rrf" else [])
            tool_hits = {k: 0 for k in TOOL_K}
            app_hits = {(g, k): 0 for g in aggs for k in APP_K}
            app_kept = {(g, k): [] for g in aggs for k in APP_K}
            miss = {(g, k): [] for g in aggs for k in APP_K}
            for i, (s, tools, gold) in enumerate(prep):
                names = [t["function"]["name"] for t in tools]
                inst = {nm.split(".")[0] for nm in names}
                r = rank[(retr, src)][i]
                for k in TOOL_K:
                    tool_hits[k] += gold <= set(r["tool"][:k])
                for g in aggs:
                    for k in APP_K:
                        keep = set(r[g][:k]) | ({"contacts"} & inst)
                        kept = {nm for nm in names if nm.split(".")[0] in keep}
                        ok = gold <= kept
                        app_hits[(g, k)] += ok
                        app_kept[(g, k)].append(len(kept) + 1)
                        if not ok:
                            miss[(g, k)].append((s["id"], queries[src][i], sorted(gold - kept)))
            n = len(prep)
            results.append({"name": name, "tool": {k: tool_hits[k] / n for k in TOOL_K},
                            "app": {f"{g}@{k}": (app_hits[(g, k)] / n, st.mean(app_kept[(g, k)])) for g in aggs for k in APP_K}})
            misses[name] = miss

    n = len(prep)
    lines = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# Dense (BGE-M3) vs BM25 tool retrieval: offline recall on train")
    out()
    out(f"Generated by `scripts/retrieval_dense.py`. {n} train samples with at least one non-sys gold tool" + (f" (first {a.limit} samples only)" if a.limit else "") + ".")
    out("Context for both retrievers: tool = name + description, app = id + name_vi + description. Query = normalised final turn, no history.")
    out(f"Today's prompt keeps every installed tool: recall 100%, mean {st.mean(len(t) + 1 for _, t, _ in prep):.1f} tools.")
    out("BGE-M3 has ~568M parameters: PhoWhisper-large (~1.55B) + Qwen3-4B (~4.02B) + BGE-M3 is ~6.1B, over the 6B system cap, so it is a measurement here, not a deployable choice as is.")
    out()
    out("## Tool level: sample recall@k (all gold tools in the top-k tools)")
    out()
    out("| retriever / text | " + " | ".join(f"@{k}" for k in TOOL_K) + " |")
    out("|---|" + "---:|" * len(TOOL_K))
    for r in results:
        out(f"| {r['name']} | " + " | ".join(f"{100 * r['tool'][k]:.1f}" for k in TOOL_K) + " |")
    out()
    out("## App level: keep the top-k apps (all their tools) + contacts; sample recall % / mean tools kept")
    out()
    out("`max` / `sum`: app score from its tools; `app`: score of the app doc itself.")
    out(f"`rrf` rows: reciprocal rank fusion (k={RRF_K}) of the dense and BM25 rankings made the same way; `all4`: RRF of dense max, dense app, BM25 max and BM25 app.")
    out()
    for g in AGGS + ["all4"]:
        out(f"| {g} | " + " | ".join(f"@{k}" for k in APP_K) + " |")
        out("|---|" + "---:|" * len(APP_K))
        for r in (x for x in results if f"{g}@1" in x["app"]):
            out(f"| {r['name']} | " + " | ".join(f"{100 * r['app'][f'{g}@{k}'][0]:.1f} / {r['app'][f'{g}@{k}'][1]:.1f}" for k in APP_K) + " |")
        out()
    out("## Verdict (ASR text, what the pipeline sees)")
    out()
    for retr in RETRS:
        r = next(x for x in results if x["name"] == f"{retr}, asr")
        bt = max(TOOL_K, key=lambda k: (r["tool"][k], -k))
        ba = max(r["app"], key=lambda x: (r["app"][x][0], -r["app"][x][1]))
        out(f"- {retr}: best tool level @{bt} {100 * r['tool'][bt]:.1f}%; best app level {ba} {100 * r['app'][ba][0]:.1f}% keeping {r['app'][ba][1]:.1f} tools.")
    d, b = (next(x for x in results if x["name"] == f"{m}, asr") for m in ("dense", "bm25"))
    gains = [100 * (d["app"][x][0] - b["app"][x][0]) for x in d["app"]]
    out(f"- dense minus BM25 over all app-level settings: {min(gains):+.1f} to {max(gains):+.1f} points; tool level @10 {100 * (d['tool'][10] - b['tool'][10]):+.1f}, @20 {100 * (d['tool'][20] - b['tool'][20]):+.1f}.")
    f = next(x for x in results if x["name"] == "rrf, asr")
    fg = [100 * (f["app"][x][0] - max(d["app"][x][0], b["app"][x][0])) for x in d["app"]]
    out(f"- RRF minus the better single retriever over the same app-level settings: {min(fg):+.1f} to {max(fg):+.1f} points.")
    best = max(max(d["app"].values(), key=lambda v: v[0])[0], max(f["app"].values(), key=lambda v: v[0])[0])
    out(f"- A filter is safe only near 100% recall (a dropped tool makes the sample score 0): best app-level recall (dense or fused) is {100 * best:.1f}%, {'safe' if best >= 0.99 else 'not safe'}.")
    if not a.limit:
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"-> {REPORT}")
    if a.misses:
        ba = max(d["app"], key=lambda x: (d["app"][x][0], -d["app"][x][1]))
        g, k = ba.split("@")
        m = misses["dense, asr"][(g, int(k))]
        print(f"\n(screen only) {min(a.misses, len(m))} of {len(m)} misses of dense, asr {ba}:")
        for sid, q, gone in m[: a.misses]:
            print(f"  {sid} | {q} | missing {', '.join(gone)}")


if __name__ == "__main__":
    main()
