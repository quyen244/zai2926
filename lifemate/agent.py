"""Stage 3: the baseline agent loop - LLM tool calling against one real ToolEnv per sample (competition rule).

B0 = gold transcript (`sample["audio_text"]`, train only, no ASR). B1 = ASR hypothesis (any split, from a
work/asr/*.jsonl file). Both call run_sample() with a different `user_text`; everything else is identical,
so the B0-vs-B1 gap on the same train samples isolates the ASR cost. B2 additionally sets `validate_first=True`
(see Stage 3 in docs/ for the "Trajectory & Baseline Pipeline" diagram this follows).
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .data import tool_env
from .llm import LLM
from .prompt import build_prompt

IDLIKE = re.compile(r"^[a-z]+(_[a-z0-9]+)*_\d+[a-z0-9_]*$")


def guard_error(ids: list[str]) -> dict:
    """The tool result the ID guard returns instead of running a call with invented IDs (also used in recovery rows)."""
    return {"ok": False, "error": f"Không tìm thấy {', '.join(ids)}: ID này chưa xuất hiện trong kết quả tra cứu nào. "
                                  "Hãy gọi tool tra cứu (search/list) để lấy đúng ID trước."}


def unseen_ids(args: dict, seen: str) -> list[str]:
    """ID arguments (key `id` / `*_id`, or a value shaped like rem_318) whose value appears nowhere in `seen`
    (the prompt plus every tool result so far): the model invented them instead of looking them up."""
    return [v for k, v in args.items()
            if isinstance(v, str) and v and (k == "id" or k.endswith("_id") or IDLIKE.match(v)) and v not in seen]


FAILURE = {"ok": False, "error": "Lỗi tạm thời từ máy chủ, vui lòng thử lại."}


def run_sample(sample: dict, split: str, user_text: str, llm: LLM, asr: dict[str, str] | None = None,
                step_cap: int = 6, validate_first: bool = False, trace: Callable[[str, dict], None] | None = None,
                keep_apps: set[str] | None = None, fail_first: str | None = None, id_guard: bool = False,
                normalise: bool = True) -> dict:
    """Run the loop on one fresh ToolEnv until the model stops calling tools or step_cap is hit.
    Returns the submission line: {"id", "tool_calls", "state_hash"} (lifemate.gold.write_submission shape).
    `trace(event, data)`, if given, sees every step: "prompt", "llm_request", "llm_reply", "parse", "validate",
    "call", "final" (scripts/trace_sample.py writes them to a text file).
    `keep_apps`, if given, is a tool-list filter: only tools of those apps (plus sys.*) are offered to the model
    (scripts/app_filter.py computes it per sample); None offers every installed app's tools, as before.
    `id_guard`: a call (not sys.*) with an ID argument never seen in the prompt or an earlier tool result is not executed
    (nothing reaches the ToolEnv) and the model gets an error asking it to look the ID up first; the row counts them.
    `fail_first`, if given, is a controlled tool failure for robustness tests: the FIRST call to that tool is not executed
    and returns FAILURE instead (nothing reaches the ToolEnv, so it is not logged); later calls run normally."""
    emit = trace or (lambda event, data: None)
    prompt = build_prompt(sample, split, user_text, asr, normalise=normalise)  # False: the RAT adapter's spoken input
    if keep_apps is not None:
        prompt = {**prompt, "tools": [t for t in prompt["tools"] if t["function"]["name"].split(".")[0] in keep_apps | {"sys"}]}
    emit("prompt", prompt)
    env = tool_env(split, sample.get("env_id", sample["id"]))  # env_id: a benchmark item reusing a train sample's state
    messages = list(prompt["messages"])
    seen = json.dumps([prompt["system"], prompt["messages"]], ensure_ascii=False) if id_guard else ""
    blocks = 0
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
            bad = unseen_ids(args, seen) if id_guard and not name.startswith("sys.") else []
            if bad:
                blocks += 1
                result = guard_error(bad)
                emit("guard", {"step": step, "name": name, "args": args, "result": result})
            elif fail_first and name == fail_first:
                fail_first = None
                result = dict(FAILURE)
            elif validate_first:
                v = env.validate(name, args)
                emit("validate", {"step": step, "name": name, "result": v})
                result = v if not v["ok"] else env.call(name, args)
            else:
                result = env.call(name, args)
            emit("call", {"step": step, "name": name, "args": args, "result": result, "logged": len(env.get_tool_calls())})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
            if id_guard:
                seen += messages[-1]["content"]
    row = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
    if id_guard:
        row["guard_blocks"] = blocks
    emit("final", row)
    return row
