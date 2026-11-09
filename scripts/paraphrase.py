"""Rewrite the final user utterance of train-fold samples with an open LLM (Kaggle Model Proxy), keep only rewrites that
pass automatic checks; the gold calls stay valid for them, so each kept rewrite becomes a new SFT row later.

  python scripts/paraphrase.py --trial 20        # 20 samples spread over R1-R4 / with and without history, printed side by side
  python scripts/paraphrase.py --all             # every train-fold sample (only after the trial was reviewed)
  python scripts/paraphrase.py --show            # print what the cache holds, call nothing; also writes the kept list

Only the train fold of work/sft/plan.train.jsonl is ever sent (never val/test, never public/private test).
Model: openai/gpt-oss-120b (Apache 2.0) through the Kaggle Benchmarks Model Proxy ($10/day AI quota); token from
`kaggle benchmarks auth`, kept in work/synth/mp.env (git-ignored) and refreshed when older than 100 minutes.
Prompts: prompts/paraphrase.{system,user}.md. Cache: work/synth/paraphrase.jsonl, one line per call, keyed by
(sample id, attempt, model, prompt hash), so a rerun calls only what is missing.
Checks per rewrite (all must hold): every "keep" phrase present (phrases of the original text equal to a gold argument
value, case-insensitive); the same digits after lifemate.normalize.normalize_text; the same resolved dates; R3 keeps a
condition word (scripts/plan_branch.py tier 1); differs from the original and from the other rewrites; 0.5x to 2x the
length; then a judge call (prompts/paraphrase_judge.*, same model) confirms the gold calls are still exactly right for it.
A sample with fewer than k kept rewrites is asked once more. reasoning_effort "low": with the default, gpt-oss spent
~790 tokens per short rewrite and the proxy queued calls (median 816 s, trial 1, 2026-11-09); low took ~1 s.
Per-sample output stays under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_samples  # noqa: E402
from lifemate.normalize import normalize_text, resolve_dates  # noqa: E402
from plan_data import scalars, meaningful, tool_kinds  # noqa: E402

OUT = PROJECT / "work" / "synth"
CACHE = OUT / "paraphrase.jsonl"
KEPT = OUT / "paraphrase.kept.jsonl"  # written by --all / --show: {"id", "n", "text", "phash"} per kept rewrite
ENV = OUT / "mp.env"
KAGGLE = PROJECT / ".venv" / "Scripts" / "kaggle.exe"
MARK = re.compile(r"\b(thì|nếu|hễ|lỡ|trường hợp|miễn là|còn không|không thì|kẻo)\b", re.I)
ASK = "sys.ask_missing_info"
_lock = threading.Lock()


def token() -> tuple[str, str]:
    """(chat completions URL, key); refreshes the proxy token when the env file is missing or older than 100 minutes."""
    with _lock:
        if not ENV.is_file() or time.time() - ENV.stat().st_mtime > 100 * 60:
            OUT.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(KAGGLE), "benchmarks", "auth", "-y", "--env-file", ENV.name], cwd=OUT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # never echo the key
        env = {k: v.strip().strip('"') for k, v in (l.strip().split("=", 1) for l in ENV.read_text(encoding="utf-8").splitlines() if "=" in l)}
    return env["MODEL_PROXY_URL"].rstrip("/") + "/openapi/chat/completions", env["MODEL_PROXY_API_KEY"]


def rule_of(sample: dict, plan_row: dict, kinds: dict) -> str:
    calls = sample["label"]["tool_calls"]
    if not calls:
        return "R4"
    if any("-> replan</plan>" in (m.get("content") or "") for m in plan_row["messages"] if m.get("train")):
        return "R3"
    if any(c["name"] == ASK and j > 0 and kinds.get(calls[j - 1]["name"]) == "read" for j, c in enumerate(calls)):
        return "R2"
    return "R1"


def keep_phrases(sample: dict) -> list[str]:
    """Phrases of the original text (its own casing) that equal a gold argument value."""
    text, out = sample["audio_text"], []
    for c in sample["label"]["tool_calls"]:
        for v in scalars(c["arguments"]):
            if isinstance(v, str) and meaningful(v):
                i = text.lower().find(v.lower())
                if i >= 0 and text[i:i + len(v)] not in out:
                    out.append(text[i:i + len(v)])
    return out


def digits(text: str) -> Counter:
    """Numbers after the spoken-form normaliser, without 1: "một" is mostly the article ("một ngày"), and normalize_text
    turns it into 1 (trial 1 rejected correct rewrites for it)."""
    return Counter(x for x in re.findall(r"\d+", normalize_text(text)) if x.lstrip("0") != "1")


def dates(text: str, now: str) -> list[str]:
    return sorted(str(d["date"]) for d in resolve_dates(normalize_text(text), now))


def check(sample: dict, rule: str, keep: list[str], cand: str, others: list[str]) -> list[str]:
    """Reasons to reject a rewrite; empty = kept."""
    orig, now = sample["audio_text"], sample["context"]["current_time"]
    bad = [f'thiếu cụm "{k}"' for k in keep if k.lower() not in cand.lower()]
    if digits(cand) != digits(orig):
        bad.append(f"số khác: {sorted(digits(orig).elements())} -> {sorted(digits(cand).elements())}")
    if dates(cand, now) != dates(orig, now):
        bad.append(f"ngày khác: {dates(orig, now)} -> {dates(cand, now)}")
    if rule == "R3" and not MARK.search(cand):
        bad.append("mất từ điều kiện")
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    if norm(cand) == norm(orig) or any(norm(cand) == norm(o) for o in others):
        bad.append("trùng câu gốc hoặc câu khác")
    if not 0.5 <= len(cand) / max(1, len(orig)) <= 2.0:
        bad.append(f"độ dài {len(cand) / max(1, len(orig)):.1f}x")
    return bad


def ask(model: str, system: str, user: str, effort: str = "low") -> dict:
    url, key = token()
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 4000, "reasoning_effort": effort}
    t0 = time.time()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=600))
            msg = ((d.get("choices") or [{}])[0] or {}).get("message") or {}
            return {"content": msg.get("content") or "", "usage": d.get("usage", {}), "sec": round(time.time() - t0, 1)}
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                return {"error": f"HTTP {e.code}: {e.read()[:200]!r}", "sec": round(time.time() - t0, 1)}
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                return {"error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 1)}
        time.sleep(5 * 2 ** attempt + random.random())
    return {"error": "unreachable"}


def parse(content: str) -> list[str]:
    m = re.search(r"\[.*\]", content, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]


def parse_objs(content: str) -> list[dict]:
    """A JSON list of objects, or one object per line (Claude answered that way once, trial 2)."""
    m = re.search(r"\[.*\]", content, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    if not items:
        for line in content.splitlines():
            try:
                items.append(json.loads(line.strip().rstrip(",")))
            except json.JSONDecodeError:
                continue
    return [x for x in items if isinstance(x, dict)]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trial", type=int, help="this many samples, spread over R1-R4 and with/without history (seed 0)")
    g.add_argument("--all", action="store_true")
    g.add_argument("--show", action="store_true", help="print the cached results only")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judges", nargs="+", default=["google/gemini-3-flash-preview", "anthropic/claude-sonnet-5@default"],
                    help="a rewrite is kept only if every judge says the gold calls still fit it. gpt-oss-120b passed all 3 known "
                         "meaning changes of trial 1 even at high effort; Gemini caught 1, Claude 2, together all 3 (2026-11-09)")
    a = ap.parse_args()

    system = (PROJECT / "prompts/paraphrase.system.md").read_text(encoding="utf-8").replace("{k}", str(a.k))
    user_t = (PROJECT / "prompts/paraphrase.user.md").read_text(encoding="utf-8")
    jsys = (PROJECT / "prompts/paraphrase_judge.system.md").read_text(encoding="utf-8")
    juser_t = (PROJECT / "prompts/paraphrase_judge.user.md").read_text(encoding="utf-8")
    pgen = hashlib.sha1((system + user_t + a.model).encode()).hexdigest()[:8]  # generation calls
    pjud = hashlib.sha1((jsys + juser_t).encode()).hexdigest()[:8]  # judge calls (the judge model is in the cache key)
    phash = f"{pgen}/{pjud}"
    rows = {r["id"]: r for r in map(json.loads, (PROJECT / "work/sft/plan.train.jsonl").read_text(encoding="utf-8").splitlines()) if r["variant"] == "gold"}
    by_id = {s["id"]: s for s in load_samples("train") if s["id"] in rows}  # train fold only
    kinds = tool_kinds()
    rule = {i: rule_of(s, rows[i], kinds) for i, s in by_id.items()}
    keep = {i: keep_phrases(s) for i, s in by_id.items()}
    cache = {}  # (id, attempt, kind) -> call row, kind = "gen" | "judge:<model>"
    if CACHE.is_file():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            kind = r.get("kind", "gen")
            if r["phash"] == (pgen if kind == "gen" else pjud):
                cache[(r["id"], r["attempt"], kind)] = r

    if a.trial:
        rng = random.Random(0)
        pick, per = [], a.trial // 4
        for R in ("R1", "R2", "R3", "R4"):
            ids = sorted(i for i in by_id if rule[i] == R)
            hist = [i for i in ids if by_id[i]["history"]]
            flat = [i for i in ids if not by_id[i]["history"]]
            rng.shuffle(hist), rng.shuffle(flat)
            pick += (hist[: per // 2] + flat[: per - per // 2])
        ids = sorted(pick)
    else:
        ids = sorted(by_id) if a.all else sorted({i for i, _, _ in cache})

    def call(i: str, attempt: int, kind: str, sys_p: str, user: str, model: str | None = None) -> dict:
        if (i, attempt, kind) in cache or a.show:
            return cache.get((i, attempt, kind), {})
        model = model or a.model
        r = {"id": i, "attempt": attempt, "kind": kind, "phash": pgen if kind == "gen" else pjud, "model": model, **ask(model, sys_p, user)}
        with _lock:
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cache[(i, attempt, kind)] = r
        return r

    def calls_text(s: dict) -> str:
        cs = s["label"]["tool_calls"]
        return "\n".join(f"{n}. {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}" for n, c in enumerate(cs, 1)) or "(không gọi tool nào)"

    def process(i: str) -> tuple[list[str], list[tuple[str, list[str]]], int]:
        """Generate, rule-check, judge; a second attempt when fewer than k rewrites survive. Returns kept, rejected, attempts."""
        s, kept, rejected, att = by_id[i], [], [], 0
        for att in (1, 2):
            g = call(i, att, "gen", system, user_t.format(text=s["audio_text"], keep="; ".join(f'"{k}"' for k in keep[i]) or "(không có)", k=a.k))
            if not g:
                break
            items = parse(g.get("content", ""))
            if not items:
                rejected.append((f"<{g.get('error') or 'không đọc được JSON'}>", ["không đọc được"]))
            passed = []
            for c in items:
                bad = check(s, rule[i], keep[i], c, kept + passed)
                (rejected.append((c, bad)) if bad else passed.append(c))
            if passed:
                ju = juser_t.format(text=s["audio_text"], calls=calls_text(s), rewrites="\n".join(f"{n}. {c}" for n, c in enumerate(passed, 1)))
                verdicts = []  # (judge short name, its verdict list)
                for jm in a.judges:
                    j = call(i, att, f"judge:{jm}", jsys, ju, jm)
                    verdicts.append((jm.split("/")[1].split("-")[0], parse_objs(j.get("content", "")) if j else []))
                for n, c in enumerate(passed):
                    bad = [f"{name}: {(v[n].get('reason') or 'không đạt') if n < len(v) else 'không chấm được'}"
                           for name, v in verdicts if not (n < len(v) and v[n].get("ok") is True)]
                    if bad:
                        rejected.append((c, bad))
                    elif len(kept) < a.k:
                        kept.append(c)
            if len(kept) >= a.k:
                break
        return kept, rejected, att

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        results = dict(zip(ids, pool.map(process, ids)))

    reasons, n_kept, cost, secs = Counter(), 0, 0, []
    for i in ids:
        kept, rejected, att = results[i]
        n_kept += len(kept)
        for _, bad in rejected:
            reasons.update(b.split(":")[0].split(' "')[0] for b in bad)
        for (cid, _, _), r in cache.items():
            if cid == i:
                c = (r.get("usage") or {}).get("cost") or {}
                cost += c.get("input_tokens_cost_nanodollars", 0) + c.get("output_tokens_cost_nanodollars", 0)
                if r.get("sec"):
                    secs.append(r["sec"])
        if a.trial or a.show:
            s = by_id[i]
            print(f"\n=== {i} | {rule[i]} | history {len(s['history'])} | {len(s['label']['tool_calls'])} calls | attempts {att}")
            print(f"  gốc  : {s['audio_text']}")
            for c in s["label"]["tool_calls"]:
                print(f"    -> {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}")
            print(f"  giữ  : {'; '.join(keep[i]) or '-'}")
            for c in kept:
                print(f"  ✓    : {c}")
            for c, bad in rejected:
                print(f"  ✗    : {c}\n         lý do: {'; '.join(bad)}")
    if not a.trial:  # the kept rewrites, for scripts/plan_data.py --para
        KEPT.write_text("".join(json.dumps({"id": i, "n": n, "text": c, "phash": phash}, ensure_ascii=False) + "\n"
                                for i in ids for n, c in enumerate(results[i][0], 1)), encoding="utf-8")
        print(f"-> {KEPT}")
    print(f"\n{len(ids)} samples | kept {n_kept} / {a.k * len(ids)} rewrites | samples with {a.k} kept: "
          f"{sum(len(results[i][0]) == a.k for i in ids)} | needed a 2nd attempt: {sum(results[i][2] == 2 for i in ids)}")
    print(f"reject reasons: {dict(reasons)}")
    print(f"proxy cost ${cost / 1e9:.4f} | calls {len(secs)}, median {sorted(secs)[len(secs) // 2] if secs else 0}s | wall {time.time() - t0:.0f}s | model {a.model} | prompts {phash}")


if __name__ == "__main__":
    main()
