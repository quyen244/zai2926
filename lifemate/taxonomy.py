"""Situation taxonomy of a sample, used to slice every evaluation.

Two kinds of tags:
  - structural tags, computed from the gold label (exact; train only): multi_action, multi_app, aggregation,
    read_then_ask, ask_now, empty, read_only, uses_history_id, long_trajectory.
  - linguistic tags, regex on a transcript (approximate; also usable on ASR output at test time): conditional,
    self_repair, negation, filler, history_reference, relative_time, money, quantity_number, dialect_central,
    dialect_south, english_word. Dialect tags only catch lexical markers, not accent.
The regexes were tuned by reading their hits on train; precision was spot-checked on a review sheet
(work/tags_review.csv), not guaranteed.
"""
from __future__ import annotations

import json
import re

LING = {
    "conditional": r"\bnếu\b|\blỡ mà\b|\btrường hợp\b|\bhễ\b|\bkhi nào\b.*\bthì\b",
    "self_repair": r"\bà (không|thôi|nhầm|mà thôi|mà không)\b|\bnhầm\b|\bthôi khỏi\b|\bý (tôi|em|anh|chị|mình) là\b|\bkhông phải\b.*\bmà là\b|\bsửa lại\b|\bđổi lại\b|\bkhoan\b",
    "negation": r"\bđừng\b|\bkhông cần\b|\bkhỏi\b|\bthôi (bỏ|không)\b",
    "filler": r"^(ờ|à|ừm|ừ|ơ|hừm|ừ thì)\b|\b(ờ|ừm|ơ)\b",
    "history_reference": r"\bcái (đó|kia|ấy|còn lại|vừa|trên|đầu|cuối|thứ \w+)\b|\b(lúc nãy|vừa rồi|vừa nãy|như trên|cái kia|người đó|chỗ đó|quán đó|bài đó|cuộc họp đó)\b|\b(nó|luôn cái)\b",
    "relative_time": r"\b(mai|ngày kia|tối nay|chiều nay|sáng nay|trưa nay|đêm nay|tuần (sau|này|tới)|tháng (sau|này|tới)|thứ (hai|ba|tư|năm|sáu|bảy|\d)|chủ nhật|cuối tuần|hôm qua)\b",
    "money": r"\b(nghìn|ngàn|triệu|tỷ|đồng|k)\b|\d+\s?k\b|\btrăm (nghìn|ngàn)\b",
    "quantity_number": r"\d|\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|rưỡi|nửa)\b",
    "dialect_central": r"\b(ni|nớ|rứa|mô|răng|tê|chừ|nờ|hỉ|bọ|mạ|o)\b",
    "dialect_south": r"\b(hông|nghen|ổng|bả|ảnh|dìa|bữa nay|hen|tui|mấy bả|xài|nhậu|đặng|chỉ giùm)\b",
    "english_word":r"\b(book|ok|oke|email|mail|app|wifi|ac|tv|playlist|check|cancel|on|off|grab|online|order|ship|voucher|meeting|zoom|note|alarm|timer|bill|smart|mode|set)\b",
}
LING_RE = {k: re.compile(v, re.IGNORECASE) for k, v in LING.items()}


def linguistic_tags(text: str) -> list[str]:
    t = " ".join(text.lower().split())
    return [k for k, r in LING_RE.items() if r.search(t)]


def structural_tags(sample: dict, kind: dict[str, str]) -> list[str]:
    calls = sample["label"]["tool_calls"]
    results = sample["label"]["tool_results"]
    tags = []
    if not calls:
        return ["empty"]
    names = [c["name"] for c in calls]
    writes = [n for n in names if kind.get(n) == "write"]
    apps = {n.split(".")[0] for n in names if not n.startswith("sys.")}
    if names[-1] == "sys.ask_missing_info":
        tags.append("ask_now" if len(names) == 1 else "read_then_ask")
    if len(set(writes)) >= 2 or len(writes) >= 2:
        tags.append("multi_action")
    if len(apps) >= 2:
        tags.append("multi_app")
    if any(names.count(n) >= 2 and kind.get(n) == "read" for n in set(names)):
        tags.append("aggregation")
    if not writes and names[-1] != "sys.ask_missing_info":
        tags.append("read_only")
    if len(calls) >= 4:
        tags.append("long_trajectory")
    hist = json.dumps(sample["history"], ensure_ascii=False)
    for i, c in enumerate(calls):
        if kind.get(c["name"]) != "write":
            continue
        for v in c["arguments"].values():
            if isinstance(v, str) and len(v) >= 3 and re.fullmatch(r"[a-z]+_[\w-]+", v):
                needle = json.dumps(v, ensure_ascii=False)
                if needle in hist and not any(needle in json.dumps(r, ensure_ascii=False) for r in results[:i]):
                    tags.append("uses_history_id")
                    break
        if "uses_history_id" in tags:
            break
    return tags
