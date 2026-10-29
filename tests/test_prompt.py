"""Stage 2.3: structural checks on lifemate.prompt (skipped when the data package is not present)."""
import json

import pytest

from lifemate.data import DATA_ROOT, load_samples
from lifemate.prompt import build_prompt

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "train" / "train.jsonl").is_file(), reason="data package not present")


@pytest.fixture(scope="module")
def sample_with_history():
    return next(s for s in load_samples("train") if s["history"])


def test_tools_match_installed_apps(sample_with_history):
    prompt = build_prompt(sample_with_history, "train", sample_with_history["audio_text"])
    tool_apps = {t["function"]["name"].split(".")[0] for t in prompt["tools"]}
    # "sys" (ask_missing_info) is always available per the catalog, regardless of installed_apps
    assert tool_apps <= set(sample_with_history["context"]["installed_apps"]) | {"sys"}
    assert tool_apps - {"sys"}  # at least one installed app actually has tools


def test_tools_always_include_ask_missing_info(sample_with_history):
    prompt = build_prompt(sample_with_history, "train", sample_with_history["audio_text"])
    names = {t["function"]["name"] for t in prompt["tools"]}
    assert "sys.ask_missing_info" in names


def test_tool_specs_are_valid_json_schema(sample_with_history):
    prompt = build_prompt(sample_with_history, "train", sample_with_history["audio_text"])
    for t in prompt["tools"]:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_messages_alternate_and_tool_calls_have_matching_results(sample_with_history):
    prompt = build_prompt(sample_with_history, "train", sample_with_history["audio_text"])
    messages = prompt["messages"]
    assert messages[-1]["role"] == "user"
    pending_ids: set[str] = set()
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert not pending_ids  # previous batch of calls must be resolved first
            pending_ids = {c["id"] for c in m["tool_calls"]}
            for c in m["tool_calls"]:
                json.loads(c["function"]["arguments"])  # arguments must round-trip as JSON
        elif m["role"] == "tool":
            assert m["tool_call_id"] in pending_ids
            pending_ids.discard(m["tool_call_id"])
    assert not pending_ids


def test_messages_require_asr_when_no_gold_text():
    public = next(s for s in load_samples("public_test") if s["history"])
    with pytest.raises(ValueError):
        build_prompt(public, "public_test", "dummy current-turn text")
