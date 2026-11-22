"""Drill-down of two error groups of a cached run (definitions: lifemate/eda.py): 1c wrong_value and 7 multi_step.

  python scripts/eda_drill.py --cache-dir <dir with one run's cache file>

1c: every (tool, argument) whose value differs from the gold in the closest call pair, split by
  - value type: id (a key ending in _id or an id-shaped value), date/time, number, list/object, code/enum
    (one short token), free text (content/body/title/...), name/query (other strings);
  - provenance, by replaying the predicted calls on a fresh ToolEnv: the gold value is in the user text, in a tool
    result the model saw, in neither (it needed a lookup the model did not make), and whether the predicted value
    came from a result (wrong entity picked) or from nowhere (invented).
7: every multi-step sample (gold 3+ calls or "nếu") that scored below full, by its first divergence from the gold
  in gold order: stopped early, hit the step cap, wrong arguments of the right tool, another tool, an extra call;
  plus the conditional ones whose predicted write tools differ from the gold's (other branch taken).
Prints aggregates only; per-sample examples go to work/eda/drill_examples.md (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_samples, tool_env  # noqa: E402
from lifemate.eda import CACHE_DIR, load_runs  # noqa: E402
from lifemate.normalize import normalize_text  # noqa: E402
from lifemate.scoring import _args, calls_match  # noqa: E402

TEXT_KEYS = {"content", "body", "title", "subject", "message", "note", "text", "description", "reason"}
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}.*)?$|^\d{1,2}:\d{2}$")
IDLIKE = re.compile(r"^[a-z]+(_[a-z0-9]+)*_\d+[a-z0-9_]*$|^[a-z]+_[a-z_]+$")
LOOKUP_PREFIX = ("search", "list", "get", "find", "read", "check")
COND = re.compile(r"\bnếu\b|không đủ|hông đủ|thiếu|mà .*thì|còn không|đủ thì")
EXAMPLES = PROJECT / "work" / "eda" / "drill_examples.md"


def norm(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return re.sub(r"\s+", " ", normalize_text(s).lower()).strip()


def value_type(key: str, g) -> str:
    if isinstance(g, bool):
        return "code/enum"
    if isinstance(g, (int, float)):
        return "number"
    if isinstance(g, (list, dict)):
        return "list/object"
    if key.endswith("_id") or key == "id" or IDLIKE.match(g):
        return "id"
    if ISO.match(g):
        return "date/time"
    if key in TEXT_KEYS:
        return "free text"
    if " " not in g.strip() and len(g) <= 14:
        return "code/enum"
    return "name/query"


def replay(sample: dict, calls: list[dict]) -> list[tuple[dict, str]]:
    """Each predicted call with the JSON result it returned, run in order on a fresh env (deterministic)."""
    env = tool_env("train", sample.get("env_id", sample["id"]))
    return [(c, json.dumps(env.call(c["name"], _args(c)), ensure_ascii=False)) for c in calls]


def user_text(sample: dict) -> str:
    parts = [sample.get("audio_text") or ""] + [t.get("audio_text") or t.get("text") or t.get("content") or "" for t in sample.get("history") or []]
    return norm(" ".join(p for p in parts if isinstance(p, str)))


def contains(hay: str, v) -> bool:
    n = norm(v)
    return bool(n) and n in hay


def fmt(c: dict) -> str:
    return f"    -> {c['name']} {json.dumps(_args(c), ensure_ascii=False)}"


def first_divergence(pred: list[dict], gold: list[dict], fz: dict, replies: int, cap: int) -> str:
    used = set()
    for gi, g in enumerate(gold):
        hit = next((i for i, p in enumerate(pred) if i not in used and calls_match(p, g, fz)), None)
        if hit is not None:
            used.add(hit)
            continue
        same = [p for i, p in enumerate(pred) if i not in used and p["name"] == g["name"]]
        if same:
            return "sai tham số (đúng tool)"
        if len(used) == len(pred):
            return "chạm step cap" if replies >= cap else "dừng sớm, thiếu bước"
        kind = "ghi" if is_write(g["name"]) else "tra cứu"
        other = [p for i, p in enumerate(pred) if i not in used and is_write(p["name"]) == is_write(g["name"])
                 and p["name"] not in {x["name"] for x in gold}]
        return f"gọi tool {kind} khác thay vì gold" if other else f"bỏ qua bước {kind} của gold"
    return "gọi thừa (gold đủ)" if len(pred) > len(used) else "đủ call, lệch hash"


def is_write(name: str) -> bool:
    return not name.startswith("sys.") and not name.split(".")[1].startswith(LOOKUP_PREFIX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--cap", type=int, default=6)
    a = ap.parse_args()
    from lifemate.data import fuzzy_params
    fz = fuzzy_params("train")
    samples = {s["id"]: s for s in load_samples("train")}
    caches = {}
    for f in a.cache_dir.glob("*.jsonl"):
        caches.update({(j := json.loads(l))["id"]: j for l in f.read_text(encoding="utf-8").splitlines() if l.strip()})
    out = ["# Drill-down examples (per sample, git-ignored)\n"]
    for (variant, f), rows in load_runs(a.cache_dir).items():
        n = len(rows)
        print(f"\n## {variant}  ({f.name}, n={n})\n")
        # ---- 1c
        vt, prov, keys, lost, mlost = Counter(), Counter(), Counter(), defaultdict(float), defaultdict(float)
        n1c = 0
        out.append(f"\n## {variant}: 1c\n")
        for r in rows:
            if "wrong_value" not in r["labels"]:
                continue
            n1c += 1
            s = samples[r["id"]]
            ut = user_text(s)
            seen, gseen = replay(s, r["pred"]), replay(s, r["gold"])
            cond_text = bool(COND.search(r["text"].lower()))
            for p, g, d in r["pairs"]:
                i = next(i for i, (c, _) in enumerate(seen) if c is p)
                gi = next(i for i, (c, _) in enumerate(gseen) if c is g)
                results = " ".join(norm(res) for _, res in seen[:i])  # what the model saw before this call
                gresults = " ".join(norm(res) for _, res in gseen[:gi])  # what the gold path had seen
                for k in d["wrong"]:
                    pv, gv = _args(p)[k], _args(g)[k]
                    t = value_type(k, gv)
                    if t == "free text":
                        m = "văn bản tự do diễn đạt khác gold"
                    elif contains(gresults, gv) and not contains(ut, gv):
                        if contains(results, gv):
                            m = "điều kiện sai trên kết quả đã thấy" if cond_text else "đọc/chọn nhầm trong kết quả đã thấy"
                        else:
                            m = "chưa tra cứu đúng nên không có giá trị"
                    elif cond_text and t in ("code/enum", "id"):
                        m = "điều kiện sai (so sánh trên kết quả)"
                    elif t == "date/time":
                        m = "tính ngày/giờ sai"
                    elif contains(ut, gv) or contains(ut, pv):
                        m = "quy ước/chuẩn hoá giá trị (mã, rút gọn)"
                    else:
                        m = "khác (suy luận nhiều bước)"
                    vt[t] += 1
                    prov[m] += 1
                    keys[f"{g['name']}.{k}"] += 1
                    lost[t] += 100 - 100 * r["score"]
                    mlost[m] += 100 - 100 * r["score"]
                    out.append(f"{r['id']} [{t}] {m} (điểm {100 * r['score']:.0f})\n{r['text']}\n  pred:\n{fmt(p)}\n  gold:\n{fmt(g)}\n")
        tot = sum(vt.values())
        print(f"### 1c: {n1c} mẫu ({100 * n1c / n:.1f}%), {tot} tham số sai giá trị\n")
        print("| Loại giá trị | Tham số | % tham số | Mẫu đó mất TB |")
        print("|---|---:|---:|---:|")
        for t, c in vt.most_common():
            print(f"| {t} | {c} | {100 * c / tot:.0f}% | {lost[t] / c:.0f} điểm |")
        print("\n| Cơ chế gây lỗi | Tham số | % tham số | Mẫu đó mất TB |")
        print("|---|---:|---:|---:|")
        for m, c in prov.most_common():
            print(f"| {m} | {c} | {100 * c / tot:.0f}% | {mlost[m] / c:.0f} điểm |")
        print("\nTham số hay sai nhất: " + ", ".join(f"`{k}` {c}" for k, c in keys.most_common(8)))
        # ---- 7
        out.append(f"\n## {variant}: nhóm 7\n")
        ms = [r for r in rows if "multi_step" in r["labels"]]
        bad = [r for r in ms if r["score"] < 1]
        kinds, klost, branch, cap_hit = Counter(), defaultdict(float), 0, 0
        cond = [r for r in bad if "nếu" in r["text"].lower()]
        for r in bad:
            replies = sum(1 for x in caches[r["id"]].get("replies", []) if x.get("tool_calls"))
            cap_hit += replies >= a.cap
            k = first_divergence(r["pred"], r["gold"], fz, replies, a.cap)
            kinds[k] += 1
            klost[k] += 100 - 100 * r["score"]
            if "nếu" in r["text"].lower():
                gw, pw = {c["name"] for c in r["gold"] if is_write(c["name"])}, {c["name"] for c in r["pred"] if is_write(c["name"])}
                if gw != pw:
                    branch += 1
            out.append(f"{r['id']} [{k}] điểm {100 * r['score']:.0f}, {replies} bước có call\n{r['text']}\n  pred:\n" +
                       "\n".join(fmt(c) for c in r["pred"]) + "\n  gold:\n" + "\n".join(fmt(c) for c in r["gold"]) + "\n")
        print(f"\n### Nhóm 7: {len(ms)} mẫu, {len(bad)} chưa đủ điểm ({sum(r['score'] == 0 for r in bad)} điểm 0); "
              f"mất {sum(100 - 100 * r['score'] for r in bad) / n:.1f} điểm trên tổng {n} mẫu\n")
        print("| Chỗ lệch đầu tiên (theo thứ tự gold) | Mẫu | % mẫu hỏng | Điểm mất TB |")
        print("|---|---:|---:|---:|")
        for k, c in kinds.most_common():
            print(f"| {k} | {c} | {100 * c / len(bad):.0f}% | {klost[k] / c:.0f} |")
        print(f"\nChạm step cap {a.cap}: {cap_hit} mẫu. Có điều kiện (\"nếu\"): {len(cond)} mẫu hỏng, trong đó {branch} có tập tool ghi khác gold (đi nhầm nhánh hoặc thiếu/thừa hành động).")
    EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES.write_text("\n".join(out), encoding="utf-8")
    print(f"\n-> {EXAMPLES}")


if __name__ == "__main__":
    main()
