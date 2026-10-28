"""Validation splits of train.

- cv5: 5 folds stratified by label type (action / read-then-ask / ask / empty) and number of gold calls.
- leave_app_out: simulates apps never seen in training (65% of public_test samples have one installed).
  A sample is in `val` when its gold calls use a held-out app, else in `train`. The held-out apps are train-only apps
  (absent from public_test), one or two per catalog group, so no app the public test uses is withheld from training.
"""
from __future__ import annotations

import random
from collections import defaultdict

HELD_OUT_APPS = ("loyalty", "fitness", "medication", "exchange_rate", "podcast_radio", "bike_rental",
                 "package_tracking", "library", "home_energy")


def label_type(calls: list[dict]) -> str:
    if not calls:
        return "empty"
    if calls[-1]["name"] == "sys.ask_missing_info":
        return "ask" if len(calls) == 1 else "read_then_ask"
    return "action"


def cv_folds(samples: list[dict], k: int = 5, seed: int = 2026) -> dict[str, int]:
    strata = defaultdict(list)
    for s in samples:
        calls = s["label"]["tool_calls"]
        strata[(label_type(calls), min(len(calls), 4))].append(s["id"])
    rng = random.Random(seed)
    fold, offset = {}, 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            fold[sid] = (i + offset) % k
        offset += len(ids)  # rotate the start so small strata do not all land in fold 0
    return fold


def leave_app_out(samples: list[dict], held_out=HELD_OUT_APPS) -> dict[str, list[str]]:
    held = set(held_out)
    out = {"train": [], "val": []}
    for s in samples:
        apps = {c["name"].split(".")[0] for c in s["label"]["tool_calls"]}
        out["val" if apps & held else "train"].append(s["id"])
    return out
