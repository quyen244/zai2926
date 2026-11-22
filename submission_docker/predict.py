"""Run the LifeMate agent on the split mounted at /data and write /result/submission.jsonl."""
import json
import time
from pathlib import Path

from runtime import Runtime, seed_everything

seed_everything(42)
rt = Runtime(Path("/data"))
cases = rt.load_cases()
print(f"preprocessing: {len(cases)} samples of {rt.split}", flush=True)
Path("/result").mkdir(parents=True, exist_ok=True)
out = Path("/result/submission.jsonl")
print(f"output will be saved in {out}", flush=True)
times = []
with open(out, "w", encoding="utf-8") as f:
    for item in cases:
        t1 = time.perf_counter()
        try:
            tool_calls, state_hash = rt.infer(item)
        except Exception as e:  # a failed sample still gets a line: no call on an untouched ToolEnv
            print(f"FAILED {item['id']}: {type(e).__name__}: {e}", flush=True)
            from lifemate.data import tool_env
            tool_calls, state_hash = [], tool_env(rt.split, item["id"], rt.root).get_state_hash()
        ms = int((time.perf_counter() - t1) * 1000)
        times.append(ms)
        f.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash, "time": ms}, ensure_ascii=False) + "\n")
        f.flush()
print(f"predict: {len(times)} samples, mean {sum(times) / max(len(times), 1):.0f} ms/sample, max {max(times, default=0)} ms", flush=True)
rt.close()
