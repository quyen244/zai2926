"""Show the ground truth of train samples: the final user turn and the gold tool calls, one line each.

  python scripts/show_gt.py train_0921                     # text + gold calls
  python scripts/show_gt.py train_0921 train_0018 --history   # also the earlier turns
  python scripts/show_gt.py train_0921 --asr --results     # also the cached PhoWhisper text and each call's gold result

Reads train only (labels exist only there). Screen only: the output is per-sample data, keep it out of the repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import PROJECT, load_samples  # noqa: E402
from request_groups import apps_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="train ids, e.g. train_0921 (a bare number such as 921 also works)")
    ap.add_argument("--history", action="store_true", help="print the earlier turns too")
    ap.add_argument("--asr", action="store_true", help="print the cached PhoWhisper text under the reference text")
    ap.add_argument("--results", action="store_true", help="print each gold call's result")
    a = ap.parse_args()

    samples = {s["id"]: s for s in load_samples("train")}
    asr = {}
    if a.asr:
        asr = {(j := json.loads(l))["audio"]: j["hyp"] for l in (PROJECT / "work/asr/phowhisper/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    for raw in a.ids:
        sid = raw if raw.startswith("train_") else f"train_{int(raw):04d}"
        s = samples.get(sid)
        if s is None:
            print(f"\n{sid}: not a train sample id")
            continue
        calls, res, h = s["label"]["tool_calls"], s["label"]["tool_results"], s["history"]
        print(f"\n- {sid}  {len(s['audio_text'].split())} từ, {len(calls)} call, app {', '.join(sorted(apps_of(calls))) or '-'}, "
              + (f"lịch sử {len(h)} lượt" if h else "không lịch sử") + f" | {s['context']['current_time']}")
        if a.history:
            for t in h:
                if t["role"] == "user":
                    print(f"    [user] {t.get('audio_text', '')}")
                else:
                    for c in t.get("tool_calls") or []:
                        print(f"    [assistant gọi] {c['name']} {c['arguments']}")
                    print(f"    [assistant] {t.get('text', '')}")
        print(f"    {'[câu cuối] ' if a.history and h else ''}{s['audio_text']}")
        if a.asr:
            print(f"    [asr] {asr.get(s['audio'], '(không có trong cache)')}")
        if not calls:
            print("    -> (gold rỗng: không gọi tool nào)")
        for i, c in enumerate(calls):
            print(f"    -> {c['name']} {c['arguments']}")
            if a.results and i < len(res):
                print(f"         = {json.dumps(res[i], ensure_ascii=False)[:300]}")


if __name__ == "__main__":
    main()
