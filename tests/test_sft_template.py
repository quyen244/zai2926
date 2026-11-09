"""lifemate.sft.history_think_template: with it, the prompt the server builds before each trained step equals that step's
left context in the training text (lifemate.sft.render); with the model's own template it does not from step 2 on.
Needs a local Qwen3 tokenizer (a training adapter folder or the HF cache); skipped otherwise. Synthetic conversation."""
import json
from pathlib import Path

import pytest

from lifemate.sft import history_think_template, render, template_messages

ADAPTER = Path(__file__).resolve().parent.parent / "work/kaggle/plan_sft_train/out/plan/adapter"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    for src in (ADAPTER, "Qwen/Qwen3-4B"):
        try:
            return AutoTokenizer.from_pretrained(str(src), local_files_only=True)
        except Exception:
            continue
    pytest.skip("no local Qwen3 tokenizer")


def call(i: int, name: str, args: dict) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


ROW = {
    "system": "Bạn là trợ lý.",
    "tools": [{"type": "function", "function": {"name": "contacts.search", "description": "tìm liên hệ",
                                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
              {"type": "function", "function": {"name": "sms.send", "description": "gửi tin",
                                                "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}}}}],
    "messages": [
        {"role": "user", "content": "chào"},
        {"role": "assistant", "content": "Chào anh."},  # an earlier turn: never trained, rendered natively in both
        {"role": "user", "content": "nhắn chị Hà là em tới muộn"},
        {"role": "assistant", "content": "<plan>contacts.search(q=\"Hà\") -> sms.send(to=$1, text=\"em tới muộn\")</plan>",
         "tool_calls": [call(1, "contacts.search", {"q": "Hà"})], "train": True},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true, \"phone\": \"0901\"}"},
        {"role": "assistant", "content": "", "tool_calls": [call(2, "sms.send", {"to": "0901", "text": "em tới muộn"})], "train": True},
        {"role": "tool", "tool_call_id": "c2", "content": "{\"ok\": true}"},
        {"role": "assistant", "content": "Dạ, em đã nhắn.", "train": True},
    ],
}


def prompts(tok, template):
    text, spans = render(ROW, tok)
    msgs = template_messages(ROW)
    steps = [i for i, m in enumerate(ROW["messages"]) if m.get("train")]
    kw = {"chat_template": template} if template else {}
    return [(text[:st], tok.apply_chat_template(msgs[: j + 1], tools=ROW["tools"], tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False, **kw)) for j, (st, _) in zip(steps, spans)]


def test_trained_template_matches_every_step(tok):
    pairs = prompts(tok, history_think_template(tok.chat_template))
    assert len(pairs) == 3
    for train_ctx, served in pairs:
        assert served == train_ctx


def test_native_template_differs_from_step_two(tok):
    pairs = prompts(tok, None)
    assert pairs[0][0] == pairs[0][1]
    assert all(a != b for a, b in pairs[1:])


def test_unknown_template_is_refused():
    with pytest.raises(AssertionError):
        history_think_template("{{ messages }}")
