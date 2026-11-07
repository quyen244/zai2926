"""One sample of the cached run, gold and predicted calls side by side, with the reasons it lost points.

  python scripts/compare_sample.py train_0007                 # B0 run (gold text)
  python scripts/compare_sample.py train_0007 --text asr      # B1 run (asr text)
  python scripts/compare_sample.py train_0007 train_0012      # several ids
  python scripts/compare_sample.py train_0007 --model qvikhr  # one model only (default: every model in the cache)

Rows pair each gold call with the predicted call of the same tool (exact match first, then the closest one).
Marks in a cell: "+" argument the gold does not have, "-" gold argument the prediction lacks, "!" different value.
Prints to the screen only. Ids per error group: scripts/error_ids.py. Full step trace: scripts/trace_sample.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import fuzzy_params  # noqa: E402
from lifemate.eda import CACHE_DIR, REASON, arg_diff, load_runs  # noqa: E402
from lifemate.scoring import _args, calls_match  # noqa: E402


def pair_rows(pred: list[dict], gold: list[dict], fz: dict) -> list[tuple[dict | None, dict | None]]:
    left, rows, partner = list(range(len(pred))), [], {}
    for exact in (True, False):
        for gi, g in enumerate(gold):
            if gi in partner:
                continue
            cand = [pi for pi in left if pred[pi]["name"] == g["name"] and (not exact or calls_match(pred[pi], g, fz))]
            if cand:
                pi = min(cand, key=lambda i: sum(len(x) for x in arg_diff(pred[i], g, fz).values()))
                partner[gi] = pi
                left.remove(pi)
    rows = [(gold[gi], pred[partner[gi]] if gi in partner else None) for gi in range(len(gold))]
    return rows + [(None, pred[pi]) for pi in left]


def cell(call: dict | None, marks: dict[str, str]) -> list[str]:
    if call is None:
        return ["-"]
    lines = [call["name"]]
    for k, v in (call.get("arguments") or {}).items():
        lines.append(f"{marks.get(k, ' ')} {k} = {json.dumps(v, ensure_ascii=False)}")
    return lines


def reasons(r: dict, rows) -> list[str]:
    out = []
    for g, p in rows:
        if g and p and not calls_match(p, g, FZ):
            d = arg_diff(p, g, FZ)
            pa, ga = _args(p), _args(g)
            for k in d["extra"]:
                out.append(f"thừa tham số: {p['name']}.{k} = {json.dumps(pa[k], ensure_ascii=False)}")
            for k in d["missing"]:
                out.append(f"thiếu tham số: {g['name']}.{k} (gold {json.dumps(ga[k], ensure_ascii=False)})")
            for k in d["wrong"]:
                out.append(f"sai giá trị: {g['name']}.{k}: gold {json.dumps(ga[k], ensure_ascii=False)}, model {json.dumps(pa[k], ensure_ascii=False)}")
        elif g and not p:
            out.append(f"gọi thiếu: {g['name']}")
        elif p and not g:
            out.append(f"gọi thừa: {p['name']}")
    for lab in r["labels"]:
        if lab in ("hash", "lookup_skip", "wrong_tool", "no_tool_wrong", "no_call", "extra_ask"):
            more = {"lookup_skip": r.get("skipped"), "wrong_tool": r.get("confusions")}.get(lab)
            out.append(REASON[lab] + (f": {', '.join(more)}" if more else ""))
    return out or ["không sai (khớp gold)"]


def show(r: dict, variant: str) -> None:
    rows = pair_rows(r["pred"], r["gold"], FZ)
    width = max(60, shutil.get_terminal_size((160, 40)).columns)
    col = (width - 7) // 2
    print("=" * width)
    print(f"{r['id']}  [{variant}]  điểm {r['score']:.2f}")
    print(f"Câu (transcript gold): {r['text']}")
    if "asr" in variant:
        print(f"Câu (ASR thô, trước normaliser): {r['asr']}")
    print("Lí do:")
    for x in reasons(r, rows):
        print(f"  - {x}")
    print()
    print(f"{'#':>2} {'gold':<{col}} | predicted")
    print(f"{'':>2} {'-' * col}-+-{'-' * col}")
    for i, (g, p) in enumerate(rows, 1):
        gm, pm = {}, {}
        if g and p:
            d = arg_diff(p, g, FZ)
            gm = {**{k: "-" for k in d["missing"]}, **{k: "!" for k in d["wrong"]}}
            pm = {**{k: "+" for k in d["extra"]}, **{k: "!" for k in d["wrong"]}}
        ok = "ok" if g and p and calls_match(p, g, FZ) else "  "
        gl = [w for x in cell(g, gm) for w in textwrap.wrap(x, col, subsequent_indent="    ") or [""]]
        pl = [w for x in cell(p, pm) for w in textwrap.wrap(x, col, subsequent_indent="    ") or [""]]
        for j in range(max(len(gl), len(pl))):
            tag = f"{i:>2}" if j == 0 else ("" if j > 1 else ok)
            print(f"{tag:>2} {gl[j] if j < len(gl) else '':<{col}} | {pl[j] if j < len(pl) else ''}")
        print(f"{'':>2} {'-' * col}-+-{'-' * col}")
    print("Dấu: + thừa so với gold, - gold có mà model thiếu, ! khác giá trị, ok = call khớp hoàn toàn")


FZ = fuzzy_params("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--text", choices=["gold", "asr"], default="gold")
    ap.add_argument("--model", default="", help="lọc theo tên model trong tên file cache (vd Qwen3, QVikhr); mặc định mọi model")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    a = ap.parse_args()
    runs = [(v, res) for (v, f), res in load_runs(a.cache_dir).items() if f".{a.text}." in f.name and a.model.lower() in f.name.lower()]
    if not runs:
        raise SystemExit(f"no {a.text} cache file matching '{a.model}' in {a.cache_dir}")
    for i in a.ids:
        for v, res in runs:
            by = {r["id"]: r for r in res}
            if i not in by:
                print(f"{i}: không có trong lượt {v}")
                continue
            show(by[i], v)


if __name__ == "__main__":
    main()
