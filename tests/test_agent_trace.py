"""run_sample's trace hook sees every step, and tracing does not change the result (skipped without the data package)."""
import json

import pytest

from lifemate.agent import run_sample
from lifemate.data import DATA_ROOT, load_samples
from lifemate.gold import build_gold

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "train" / "train.jsonl").is_file(), reason="data package not present")


class Gold:
    def __init__(self, calls):
        self.calls = list(calls)

    def chat_tools(self, system, messages, tools, **_):
        if not self.calls:
            return {"role": "assistant", "content": "xong", "tool_calls": []}
        c = self.calls.pop(0)
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "g", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]}


def test_trace_events_and_same_result():
    s = next(x for x in load_samples("train") if x["id"] == "train_0001")
    events = []
    row = run_sample(s, "train", s["audio_text"], Gold(s["label"]["tool_calls"]), validate_first=True,
                     trace=lambda e, d: events.append(e))
    plain = run_sample(s, "train", s["audio_text"], Gold(s["label"]["tool_calls"]), validate_first=True)
    n = len(s["label"]["tool_calls"])
    assert events[0] == "prompt" and events[-1] == "final"
    assert events.count("llm_request") == n + 1 and events.count("call") == n and events.count("validate") == n
    assert row == plain and row["state_hash"] == build_gold("train")["train_0001"]["state_hash"]


def test_env_id_reuses_another_samples_state():
    """A benchmark item (scripts/build_hard_bench.py) carries its own id and runs on a train sample's state via env_id."""
    s = next(x for x in load_samples("train") if x["id"] == "train_0001")
    item = {**s, "id": "hard_x_001", "env_id": "train_0001"}
    row = run_sample(item, "train", s["audio_text"], Gold(s["label"]["tool_calls"]))
    assert row["id"] == "hard_x_001" and row["state_hash"] == build_gold("train")["train_0001"]["state_hash"]
