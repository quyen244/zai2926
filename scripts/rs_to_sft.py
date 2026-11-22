"""Rebuild high-reward B2 rejection-sampling trajectories as full-tool SFT rows."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifemate.data import catalog_tools, load_samples, tool_env, zaic  # noqa: E402
from lifemate.prompt import build_prompt  # noqa: E402
AUDIT = ROOT / "work/kaggle/rs_audit_artifacts"
PROMPTS = ROOT / "work/rs_b2/prompts.rs.jsonl"
SELECTED = AUDIT / "rs_selected.jsonl"
SFT_BASE = ROOT / "work/sft/traj.train.jsonl"
VAL_BASE = ROOT / "work/sft/traj.val.jsonl"
OUT = ROOT / "work/rs_sft"
MIN_SCORE = 80.0  # scorer emits 0-100; the rollout job's 0.8 threshold was too permissive.
DATE_HINT = re.compile(r"\n\(Ngày cụ thể cho các mốc thời gian nói tới:.*\)\s*$", re.S)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    prompts = read_jsonl(PROMPTS)
    selected = read_jsonl(SELECTED)
    samples = {row["id"]: row for row in load_samples("train")}
    prompt_by_index = {i: row for i, row in enumerate(prompts)}
    if len(prompts) != 2234 or len(prompt_by_index) != len(prompts):
        raise SystemExit(f"unexpected prompt bundle size: {len(prompts)}")
    if len(selected) != 752 or len({row["prompt_index"] for row in selected}) != len(selected):
        raise SystemExit("selected RS file is incomplete or contains multiple rows per prompt")

    catalog = catalog_tools("train")
    tools_api = zaic("train")
    source_rows = read_jsonl(SFT_BASE)
    source_by_id = {}
    for row in source_rows:
        if row.get("variant") == "gold":
            source_by_id.setdefault(row["id"], row)

    val_rows = read_jsonl(VAL_BASE)
    val_ids = {row["id"] for row in val_rows}
    prompt_ids = {row["id"] for row in prompts}
    overlap = prompt_ids & val_ids
    if overlap:
        raise SystemExit(f"RS train prompts overlap validation IDs: {sorted(overlap)[:5]}")

    out_rows = []
    excluded = Counter()
    for picked in selected:
        if picked["score"] < MIN_SCORE:
            excluded["score_below_80"] += 1
            continue
        idx = picked["prompt_index"]
        prompt = prompt_by_index.get(idx)
        if prompt is None or prompt["id"] != picked["id"]:
            raise SystemExit(f"prompt mapping mismatch at index {idx}")
        sample = samples[picked["id"]]
        allowed = {name for app in sample["context"]["installed_apps"].values() for name in app.get("tools", [])}
        allowed.add("sys.ask_missing_info")
        schema = tools_api.schema_validate(picked["tool_calls"], catalog, allowed)
        if any(not result["ok"] for result in schema):
            excluded["schema_invalid"] += 1
            continue

        env = tool_env("train", picked["id"])
        result_pairs = []
        for call in picked["tool_calls"]:
            result = env.call(call["name"], call.get("arguments") or {})
            result_pairs.append((call, result))
        if env.get_state_hash() != picked["state_hash"]:
            raise SystemExit(f"state replay mismatch: {picked['id']} prompt {idx}")

        base = source_by_id.get(picked["id"])
        if base is None:
            raise SystemExit(f"missing base response for {picked['id']}")
        final_texts = [m.get("content") for m in base["messages"]
                       if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls")]
        if not final_texts:
            raise SystemExit(f"missing assistant finish text for {picked['id']}")

        # prompts.rs.jsonl stores already-normalized text with the relative-date hint. Strip that suffix before
        # build_prompt, which adds the hint itself, so training sees the same one-hint format as inference.
        raw_user_text = DATE_HINT.sub("", prompt["user_text"]).strip()
        built = build_prompt(sample, "train", raw_user_text)
        messages = list(built["messages"])
        for call_i, (call, result) in enumerate(result_pairs):
            call_id = f"rs_{idx}_{call_i}"
            messages.append({"role": "assistant", "content": None, "train": True, "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": call["name"],
                             "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)},
            }]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": final_texts[-1], "train": True})
        out_rows.append({
            "id": f"{picked['id']}_rs_{idx}",
            "source_id": picked["id"],
            "source": prompt["source"],
            "source_variant": prompt["variant"],
            "prompt_index": idx,
            "candidate": picked["candidate"],
            "score": picked["score"],
            "split": "train",
            "system": built["system"],
            "tools": built["tools"],
            "messages": messages,
        })

    v1_rows = []
    for source in source_rows:
        row = dict(source)
        row["source_id"] = source["id"]
        row["id"] = f"{source['id']}_v1_{source['variant']}"
        row["mix_source"] = "v1_original"
        v1_rows.append(row)
    mixed_rows = out_rows + v1_rows

    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {
        "rs_only.train.jsonl": out_rows,
        "rs_only.val.jsonl": val_rows,
        "rs_plus_v1.train.jsonl": mixed_rows,
        "rs_plus_v1.val.jsonl": val_rows,
    }
    for name, rows in datasets.items():
        (OUT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    stats = {
        "selected_job_rows": len(selected),
        "score_threshold": MIN_SCORE,
        "train_rows": len(out_rows),
        "v1_replay_rows": len(v1_rows),
        "mixed_rows": len(mixed_rows),
        "excluded": dict(excluded),
        "validation_rows": len(val_rows),
        "train_validation_id_overlap": 0,
        "mean_score": round(sum(row["score"] for row in out_rows) / max(1, len(out_rows)), 4),
        "tool_counts": dict(Counter(len(row["tools"]) for row in out_rows)),
        "state_hashes_replayed": len(out_rows),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
