"""Local re-implementation of the official LifeMate metric.

Per sample (official rules):
  - gold tool_calls empty: 1 if the submitted list is also empty, else 0.
  - otherwise 0 if the submitted state_hash differs from the gold one or no call matches (M = 0),
    else F1 = 2PR / (P + R) with P = M / (M + U), R = M / |G|.
  - A submitted call matches a gold call when the tool name is the same, every non-fuzzy argument is identical and every
    fuzzy argument (catalog x-match: fuzzy) has token-F1 > 0.6 after lower-casing and Unicode NFC. Each call matches at
    most one call; order is ignored (maximum bipartite matching). Rejected calls count in U.
Final score = 100 * mean(score).

Assumptions where the rules are silent (kept explicit so they can be changed in one place):
  - Arguments whose value is empty (None, "", [], {}) are treated as absent, as zaic_tools.validate does.
  - "Identical" is Python equality of the JSON values (so 26 == 26.0, list order matters, strings are case-sensitive).
  - Fuzzy tokens are whitespace-separated words after lower + NFC; token-F1 is the bag-of-tokens F1 (SQuAD style).
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

EMPTY = (None, "", [], {})
FUZZY_THRESHOLD = 0.6


def _tokens(s) -> list[str]:
    return unicodedata.normalize("NFC", str(s).lower()).split()


def token_f1(pred, gold) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def _args(call: dict) -> dict:
    a = call.get("arguments") or {}
    return {k: v for k, v in a.items() if v not in EMPTY} if isinstance(a, dict) else {}


def calls_match(pred: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    fz = fuzzy.get(gold.get("name"), set())
    pa, ga = _args(pred), _args(gold)
    if set(pa) != set(ga):
        return False
    for k, gv in ga.items():
        if k in fz:
            if token_f1(pa[k], gv) <= FUZZY_THRESHOLD:
                return False
        elif pa[k] != gv:
            return False
    return True


def max_matching(pred: list[dict], gold: list[dict], fuzzy: dict[str, set[str]]) -> int:
    """Size of a maximum one-to-one matching between submitted and gold calls (augmenting paths)."""
    adj = [[j for j, g in enumerate(gold) if calls_match(p, g, fuzzy)] for p in pred]
    owner = [-1] * len(gold)

    def augment(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, [False] * len(gold)) for i in range(len(pred)))


@dataclass
class SampleScore:
    score: float
    hash_ok: bool
    matched: int
    unmatched: int
    n_gold: int
    n_pred: int
    precision: float = field(default=0.0)
    recall: float = field(default=0.0)


def score_sample(pred_calls: list[dict], pred_hash: str, gold_calls: list[dict], gold_hash: str,
                 fuzzy: dict[str, set[str]]) -> SampleScore:
    pred_calls, gold_calls = pred_calls or [], gold_calls or []
    hash_ok = pred_hash == gold_hash
    if not gold_calls:
        return SampleScore(float(not pred_calls), hash_ok, 0, len(pred_calls), 0, len(pred_calls),
                           float(not pred_calls), 1.0)
    m = max_matching(pred_calls, gold_calls, fuzzy)
    u = len(pred_calls) - m
    p = m / (m + u) if m + u else 0.0
    r = m / len(gold_calls)
    f1 = 2 * p * r / (p + r) if m else 0.0
    return SampleScore(f1 if hash_ok and m else 0.0, hash_ok, m, u, len(gold_calls), len(pred_calls), p, r)


def score_split(preds: dict[str, dict], golds: dict[str, dict], fuzzy: dict[str, set[str]]) -> tuple[float, dict[str, SampleScore]]:
    """preds / golds: {id: {"tool_calls": [...], "state_hash": "..."}}. A missing prediction counts as an empty submission
    with no hash (scores 1 only on an empty gold). Returns (final score 0-100, per-sample scores)."""
    per = {}
    for sid, g in golds.items():
        p = preds.get(sid, {"tool_calls": [], "state_hash": None})
        per[sid] = score_sample(p.get("tool_calls") or [], p.get("state_hash"), g["tool_calls"], g["state_hash"], fuzzy)
    final = 100.0 * sum(s.score for s in per.values()) / len(per) if per else 0.0
    return final, per
