"""Apply the selected E5 top-7 app filter to RAT gold + paraphrase SFT rows."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifemate.app_filter import AppFilter  # noqa: E402

SOURCE = ROOT / "work/sft_rat_toolonly_gold_para"
OUT = ROOT / "work/sft_rat_toolonly_gold_para_top7"
E5 = ROOT / "work/kaggle/embed_train_q/out/app_embedder_e5"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def app_of(tool: dict) -> str:
    return tool["function"]["name"].split(".", 1)[0]


def main() -> None:
    if not (E5 / "model.safetensors").is_file():
        raise SystemExit(f"missing selected E5 checkpoint: {E5}")
    filt = AppFilter(E5, device="cpu")
    counts = Counter()
    output = {}
    omitted = []
    for split in ("train", "val"):
        result = []
        for row in rows(SOURCE / f"traj.{split}.jsonl"):
            user_text = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
            installed = sorted({app_of(t) for t in row["tools"] if app_of(t) != "sys"})
            keep = filt.keep(user_text, installed) | {"sys"}
            tools = [t for t in row["tools"] if app_of(t) in keep]
            gold_tools = {c["function"]["name"] for m in row["messages"] if m.get("train") for c in m.get("tool_calls", [])}
            offered = {t["function"]["name"] for t in tools}
            if not gold_tools <= offered:
                omitted.append((split, row["id"], row["variant"], sorted(gold_tools - offered)))
                continue
            result.append({**row, "tools": tools})
            counts[f"{split}_{row['variant']}"] += 1
            counts[f"{split}_tools_before"] += len(row["tools"])
            counts[f"{split}_tools_after"] += len(tools)
        output[split] = result
    if omitted:
        # No gold label is silently trained against a prompt that hides its tool.
        print(json.dumps({"missing_gold_tool_count": len(omitted), "examples": omitted[:10]}, ensure_ascii=False, indent=2))
        raise SystemExit("top-7 omitted a gold tool; resolve before packaging")
    OUT.mkdir(parents=True, exist_ok=True)
    for split, result in output.items():
        (OUT / f"traj.{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result), encoding="utf-8")
    summary = {"source": str(SOURCE.relative_to(ROOT)), "filter": "selected E5 top-7 + contacts + sys",
               "rows": {split: len(result) for split, result in output.items()}, "counts": dict(counts),
               "missing_gold_tools": 0}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
