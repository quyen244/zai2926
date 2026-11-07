"""Stage 3: the baseline agent loop - LLM tool calling against one real ToolEnv per sample (competition rule).

B0 = gold transcript (`sample["audio_text"]`, train only, no ASR). B1 = ASR hypothesis (any split, from a
work/asr/*.jsonl file). Both call run_sample() with a different `user_text`; everything else is identical,
so the B0-vs-B1 gap on the same train samples isolates the ASR cost. B2 additionally sets `validate_first=True`
(see Stage 3 in docs/ for the "Trajectory & Baseline Pipeline" diagram this follows).
"""
from __future__ import annotations

import json
from typing import Callable

from .data import tool_env
from .llm import LLM
from .prompt import build_prompt


def run_sample(sample: dict, split: str, user_text: str, llm: LLM, asr: dict[str, str] | None = None,
                step_cap: int = 6, validate_first: bool = False, trace: Callable[[str, dict], None] | None = None) -> dict:
    """Run the loop on one fresh ToolEnv until the model stops calling tools or step_cap is hit.
    Returns the submission line: {"id", "tool_calls", "state_hash"} (lifemate.gold.write_submission shape).
    `trace(event, data)`, if given, sees every step: "prompt", "llm_request", "llm_reply", "parse", "validate",
    "call", "final" (scripts/trace_sample.py writes them to a text file)."""
    emit = trace or (lambda event, data: None)
    prompt = build_prompt(sample, split, user_text, asr)
    emit("prompt", prompt)
    env = tool_env(split, sample.get("env_id", sample["id"]))  # env_id: a benchmark item reusing a train sample's state
    messages = list(prompt["messages"])
    for step in range(step_cap):
        emit("llm_request", {"step": step, "system": prompt["system"], "messages": messages, "tools": prompt["tools"]})
        reply = llm.chat_tools(prompt["system"], messages, prompt["tools"])
        emit("llm_reply", {"step": step, "reply": reply})
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
            emit("parse", {"step": step, "name": name, "raw": c["function"]["arguments"], "args": args})
            if validate_first:
                v = env.validate(name, args)
                emit("validate", {"step": step, "name": name, "result": v})
                result = v if not v["ok"] else env.call(name, args)
            else:
                result = env.call(name, args)
            emit("call", {"step": step, "name": name, "args": args, "result": result, "logged": len(env.get_tool_calls())})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
    row = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
    emit("final", row)
    return row
