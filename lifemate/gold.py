"""Gold answers for labelled splits and submission I/O.

The gold state_hash of a train sample is not shipped; it is obtained by replaying label.tool_calls in a fresh ToolEnv
(the organisers score the same way: replay with the split's zaic_tools on the original state.json).
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import DATA_ROOT, load_catalog, load_samples, tool_env, zaic


def replay(split: str, sample_id: str, calls: list[dict], root: Path = DATA_ROOT):
    """Run `calls` on a fresh env; returns (env, results)."""
    env = tool_env(split, sample_id, root)
    results = [env.call(c["name"], c.get("arguments") or {}) for c in calls]
    return env, results


def build_gold(split: str = "train", root: Path = DATA_ROOT) -> dict[str, dict]:
    """{id: {"tool_calls", "state_hash", "replay_ok"}}; replay_ok = replayed state equals label.final_state for every
    changed app and every replayed result equals label.tool_results."""
    gold = {}
    for s in load_samples(split, root):
        lab = s["label"]
        env, results = replay(split, s["id"], lab["tool_calls"], root)
        state = env.get_state()
        ok = all(state.get(a) == v for a, v in lab["final_state"].items()) and results == lab["tool_results"]
        gold[s["id"]] = {"tool_calls": lab["tool_calls"], "state_hash": env.get_state_hash(), "replay_ok": ok}
    return gold


def initial_hash(split: str, sample_id: str, root: Path = DATA_ROOT) -> str:
    return tool_env(split, sample_id, root).get_state_hash()


def read_submission(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def write_submission(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "tool_calls": r["tool_calls"], "state_hash": r["state_hash"]},
                               ensure_ascii=False) + "\n")


def validate_submission(path: str | Path, split: str, root: Path = DATA_ROOT) -> list[str]:
    """Format problems of a submission file (empty list = valid): UTF-8 JSONL, exactly one line per sample of the split,
    the three fields with the right types, and every call schema-valid for the sample's installed apps.
    Schema-invalid calls are legal in a submission (they are logged by ToolEnv) but are always unmatched, so they are
    reported as warnings prefixed with 'warn:'."""
    problems: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e:
        return [f"not UTF-8: {e}"]
    samples = {s["id"]: s for s in load_samples(split, root)}
    seen: set[str] = set()
    zt = zaic(split, root)
    catalog = load_catalog(split, root)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            problems.append(f"line {n}: empty line")
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e})")
            continue
        if set(r) != {"id", "tool_calls", "state_hash"}:
            problems.append(f"line {n}: keys {sorted(r)} != ['id', 'state_hash', 'tool_calls']")
        sid = r.get("id")
        if sid not in samples:
            problems.append(f"line {n}: unknown id {sid!r}")
            continue
        if sid in seen:
            problems.append(f"line {n}: duplicate id {sid}")
        seen.add(sid)
        if not isinstance(r.get("state_hash"), str) or len(r["state_hash"]) != 64:
            problems.append(f"line {n}: state_hash must be a 64-char sha256 hex string")
        calls = r.get("tool_calls")
        if not isinstance(calls, list) or not all(isinstance(c, dict) and isinstance(c.get("arguments", {}), dict) for c in calls):
            problems.append(f"line {n}: tool_calls must be a list of {{name, arguments}} objects")
            continue
        allowed = {t for a in samples[sid]["context"]["installed_apps"].values() for t in a.get("tools", [])} | {"sys.ask_missing_info"}
        for i, v in enumerate(zt.schema_validate(calls, catalog, allowed)):
            if not v["ok"]:
                problems.append(f"warn: {sid} call {i} {v['name']}: {v['error']}")
    missing = set(samples) - seen
    if missing:
        problems.append(f"{len(missing)} sample(s) missing, e.g. {sorted(missing)[:3]}")
    return problems
