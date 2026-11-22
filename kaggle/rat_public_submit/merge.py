"""Join the four RAT public_test shards (kaggle/rat_public_submit*, outputs in work/kaggle/rat_public_submit_s{k}/) into
work/kaggle/rat_public_submit/submission.jsonl in public_test.jsonl order; a sample missing from every shard gets an empty
answer (no call, untouched state hash). Then validate with: python -m zaic_tools --split public_test --root .. --validate <file>
(run inside data/train+public_test/public_test)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lifemate.data import load_samples, tool_env  # noqa: E402

rows = {}
for k in range(4):
    f = ROOT / "work" / "kaggle" / f"rat_public_submit_s{k}" / f"submission_shard{k}.jsonl"
    got = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.is_file() else []
    print(f"shard {k}: {len(got)} lines" + ("" if got else f" (missing {f})"))
    rows.update({r["id"]: r for r in got})
order = [s["id"] for s in load_samples("public_test")]
lines = [rows.get(i) or {"id": i, "tool_calls": [], "state_hash": tool_env("public_test", i).get_state_hash()} for i in order]
out = ROOT / "work" / "kaggle" / "rat_public_submit" / "submission.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
print(f"{out}: {len(lines)} lines, {sum(i in rows for i in order)} from shards, {sum(not l['tool_calls'] for l in lines)} empty answers")
