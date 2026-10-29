"""Stage 3: the baseline agent loop - LLM tool calling against one real ToolEnv per sample (competition rule).

B0 = gold transcript (`sample["audio_text"]`, train only, no ASR). B1 = ASR hypothesis (any split, from a
work/asr/*.jsonl file). Both call run_sample() with a different `user_text`; everything else is identical,
so the B0-vs-B1 gap on the same train samples isolates the ASR cost. B2 additionally sets `validate_first=True`
(see Stage 3 in docs/ for the "Trajectory & Baseline Pipeline" diagram this follows).
"""
from __future__ import annotations

import json

from .data import tool_env
from .llm import LLM
from .prompt import build_prompt


def run_sample(sample: dict, split: str, user_text: str, llm: LLM, asr: dict[str, str] | None = None,
                step_cap: int = 6, validate_first: bool = False) -> dict:
    """Run the loop on one fresh ToolEnv until the model stops calling tools or step_cap is hit.
    Returns the submission line: {"id", "tool_calls", "state_hash"} (lifemate.gold.write_submission shape)."""
    prompt = build_prompt(sample, split, user_text, asr)
    env = tool_env(split, sample["id"])
    messages = list(prompt["messages"])
    for _ in range(step_cap):
        reply = llm.chat_tools(prompt["system"], messages, prompt["tools"])
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if validate_first:
                v = env.validate(name, args)
                result = v if not v["ok"] else env.call(name, args)
            else:
                result = env.call(name, args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
    return {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
