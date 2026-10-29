"""Stage 2.3: build the chat-completions prompt (system + tools + messages) for one sample.

Design (see `scripts/prompt_mock.py` to inspect a real example):

- `tools`: OpenAI function-calling specs, one per tool of the apps installed in THIS sample only
  (not the whole catalog: cuts prompt size and matches what the sample's ToolEnv actually offers).
- `system`: fixed behaviour rules (same text for every sample, see SYSTEM_INSTRUCTIONS) followed by
  a per-sample context block (current time, timezone, location, user profile). Context is per-sample
  data, so it cannot be a separate static/cacheable block; it is still conceptually "system", just not
  textually fixed.
- `messages`: the turn history replayed as a real multi-turn tool-calling conversation (user text,
  assistant tool_calls, "tool"-role results, assistant reply), then the final user turn to answer now.
  The text of every user turn goes through `build_user_text` (spoken-form normaliser + relative-date
  hints from `lifemate.normalize`), so numbers/money/clock times are already in a form close to what
  the gold arguments use, and the model does not have to do date arithmetic from the raw transcript.

Text source (ASR vs gold transcript) is injected by the caller via `user_text` / `history_text`, so the
same builder serves both the oracle baseline (B0, gold `audio_text`) and the ASR baseline (B1); only the
text source differs, not the prompt shape.
"""
from __future__ import annotations

import json
from datetime import datetime

from lifemate.data import DATA_ROOT, load_catalog
from lifemate.normalize import normalize_text, resolve_dates

SYSTEM_INSTRUCTIONS = """\
Bạn là LifeMate, trợ lý giọng nói điều khiển các app đã cài trên điện thoại người dùng qua tool calling.

Quy tắc:
- Chỉ gọi tool trong danh sách "tools" được cung cấp (đúng app đã cài cho người dùng này).
- Không tự đặt ra tool hoặc tham số không có trong schema.
- Nếu yêu cầu thiếu tham số bắt buộc, hoặc một thực thể (tên người, phòng, thiết bị...) có thể khớp \
nhiều đối tượng, hỏi lại người dùng bằng lời thay vì đoán hoặc tự chọn một giá trị.
- Nếu không app nào đã cài làm được yêu cầu, hoặc người dùng chỉ trò chuyện không cần hành động, trả lời \
bằng lời, không gọi tool nào.
- Nếu người dùng tự hủy hoặc đổi ý trong chính lượt nói, không gọi tool nào cho phần đã hủy.
- Dùng giờ hiện tại và hồ sơ người dùng ở dưới để suy ra tham số ngầm định (ví dụ "về nhà" -> địa chỉ nhà \
trong hồ sơ, "giờ hiện tại" cho các mốc thời gian tương đối).
"""


def build_user_text(raw_text: str, now: str | datetime) -> str:
    """Spoken-form normaliser + a relative-date hint line, applied to every user turn (history and current)."""
    text = normalize_text(raw_text)
    dates = resolve_dates(text, now)
    if not dates:
        return text
    hints = "; ".join(f'"{d["phrase"]}" = {d["date"]}' for d in dates)
    return f"{text}\n(Ngày cụ thể cho các mốc thời gian nói tới: {hints})"


def build_context_block(ctx: dict) -> str:
    profile = ctx.get("user_profile") or {}
    lines = [
        f"Giờ hiện tại: {ctx['current_time']} ({ctx['timezone']})",
        f"Vị trí hiện tại: {ctx['location']['lat']}, {ctx['location']['lng']}",
        "Hồ sơ người dùng: " + json.dumps(profile, ensure_ascii=False),
    ]
    return "\n".join(lines)


def build_tools(split: str, installed_apps: dict, root=DATA_ROOT) -> list[dict]:
    """OpenAI-style function specs, one per tool of the installed apps only."""
    tools = []
    for app in load_catalog(split, root)["apps"]:
        if app["id"] not in installed_apps:
            continue
        for t in app["tools"]:
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            }})
    return tools


def _tool_call_messages(turn: dict, idx: int) -> list[dict]:
    """One assistant message carrying tool_calls, then one tool-role message per result."""
    calls = turn.get("tool_calls") or []
    ids = [f"call_{idx}_{i}" for i in range(len(calls))]
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
        for cid, c in zip(ids, calls)
    ]}]
    for cid, r in zip(ids, turn.get("tool_results") or []):
        msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(r, ensure_ascii=False)})
    if turn.get("text"):
        msgs.append({"role": "assistant", "content": turn["text"]})
    return msgs


def build_messages(sample: dict, user_text: str, asr: dict[str, str] | None = None) -> list[dict]:
    """Replay `sample["history"]` as real chat turns, then append the current (already-normalised) user turn.

    `asr`: optional {audio relative path -> ASR hypothesis}, e.g. loaded from a work/asr/*.jsonl file.
    Falls back to the turn's own gold `audio_text` when a path is not in `asr` (or when `asr` is None);
    public_test history turns have no gold text at all, so `asr` is required to cover every clip there.
    """
    now = sample["context"]["current_time"]
    asr = asr or {}
    messages: list[dict] = []
    for i, turn in enumerate(sample["history"]):
        if turn["role"] == "user":
            raw = asr.get(turn["audio"], turn.get("audio_text"))
            if raw is None:
                raise ValueError(f"no text for history turn {i} (audio={turn['audio']}): pass it via `asr`")
            messages.append({"role": "user", "content": build_user_text(raw, now)})
        else:
            if turn.get("tool_calls"):
                messages += _tool_call_messages(turn, i)
            elif turn.get("text"):
                messages.append({"role": "assistant", "content": turn["text"]})
    messages.append({"role": "user", "content": build_user_text(user_text, now)})
    return messages


def build_prompt(sample: dict, split: str, user_text: str, asr: dict[str, str] | None = None, root=DATA_ROOT) -> dict:
    """{"system", "tools", "messages"} ready to send as a chat-completions request (system is a separate string,
    as the OpenAI-compatible client in lifemate.llm expects; prepend it as the first message for other APIs)."""
    ctx = sample["context"]
    system = SYSTEM_INSTRUCTIONS + "\n" + build_context_block(ctx)
    return {
        "system": system,
        "tools": build_tools(split, ctx["installed_apps"], root),
        "messages": build_messages(sample, user_text, asr),
    }
