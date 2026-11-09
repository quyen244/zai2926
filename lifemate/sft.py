"""SFT examples for the plan ablation: render a work/sft/*.jsonl row with the Qwen3 chat template and mask the loss.

The row's messages are the agent prompt (system, history replayed, final user turn) followed by the trained turns
(scripts/plan_data.py marks each trained assistant message with "train": true). The whole conversation is rendered once
with the tokenizer's own chat template, tools included and tool-call arguments as objects (what vLLM renders at
inference), enable_thinking=False. Loss is on the trained assistant messages only: the tokens after
ASSISTANT + THINK (the empty think block, inserted where the template leaves it out), up to and including "<|im_end|>".
Because every trained turn keeps its think block, a later step's left context has the earlier steps WITH it, which the
model's own template never renders: serve the adapters with history_think_template(), which renders exactly this text. User text, tool results, the system prompt, the tool schemas and history turns get label -100.
"""
from __future__ import annotations

import json

IGNORE = -100
ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>"
THINK = "<think>\n\n</think>\n\n"


NATIVE_BRANCH = "{%- if loop.last or (not loop.last and reasoning_content) %}"


def history_think_template(native: str) -> str:
    """The Qwen3 chat template with one change: an assistant message of the current turn that is already in the history
    keeps its (empty) think block, as render() trains it. The native template drops an empty think block from every
    such message except the last, so from the agent's second step on, the text vLLM builds differed from the training
    text by 6 tokens per earlier step (701 of 1010 trained test turns, found 2026-11-09). Serving the adapters with this
    template makes every generation prompt equal to its training left context (tests/test_sft_template.py)."""
    assert native.count(NATIVE_BRANCH) == 1, "unexpected Qwen3 chat template"
    return native.replace(NATIVE_BRANCH, "{%- if true %}")


def template_messages(row: dict) -> list[dict]:
    msgs = [{"role": "system", "content": row["system"]}]
    for m in row["messages"]:
        m = {k: v for k, v in m.items() if k != "train"}
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": a if isinstance(a := c["function"]["arguments"], dict) else json.loads(a or "{}")}} for c in m["tool_calls"]]
        if m.get("content") is None:
            m["content"] = ""
        msgs.append(m)
    return msgs


def render(row: dict, tok) -> tuple[str, list[tuple[int, int]]]:
    """Full text and the character spans that carry loss (one per trained assistant message, in order)."""
    text = tok.apply_chat_template(template_messages(row), tools=row["tools"], tokenize=False, enable_thinking=False)
    n_train = sum(1 for m in row["messages"] if m.get("train"))
    starts, i = [], 0
    while (i := text.find(ASSISTANT, i)) != -1:
        starts.append(i)
        i += len(ASSISTANT)
    # At inference every generation starts right after ASSISTANT + THINK (enable_thinking=False adds the empty think
    # block to the generation prompt), but the template keeps that block only on the LAST assistant message. Insert it
    # before every trained message so each trained turn sees the same left context as the step that generates it.
    for s in reversed(starts[-n_train:]):
        a = s + len(ASSISTANT)
        if not text.startswith(THINK, a):
            text = text[:a] + THINK + text[a:]
    spans, i = [], 0
    starts = []
    while (i := text.find(ASSISTANT, i)) != -1:
        starts.append(i)
        i += len(ASSISTANT)
    for s in starts[-n_train:]:  # trained turns are the last assistant messages of the conversation
        a = s + len(ASSISTANT) + len(THINK)
        e = text.index(END, a) + len(END)
        spans.append((a, e))
    return text, spans


def encode(row: dict, tok, max_len: int) -> dict:
    """input_ids / labels for one row; raises if the row is longer than max_len (plan_data.py already checked)."""
    text, spans = render(row, tok)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        raise ValueError(f"{row['id']} {row['variant']}: {len(ids)} tokens > {max_len}")
    labels = [IGNORE] * len(ids)
    for k, (a, b) in enumerate(offs):
        if any(a >= s and b <= e and b > a for s, e in spans):
            labels[k] = ids[k]
    return {"input_ids": ids, "labels": labels, "n_tokens": len(ids), "n_supervised": sum(x != IGNORE for x in labels)}
