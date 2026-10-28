"""Paths and loaders for the competition data package.

The data lives in data/train+public_test/<split>/ (git-ignored). Only the jsonl, the tools catalog, the audio and the
split's own zaic_tools are read here; <split>_environment/ is never opened directly (competition rule), the environment
is reached only through zaic_tools.ToolEnv.
"""
from __future__ import annotations

import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "data" / "train+public_test"
SPLITS = ("train", "public_test", "private_test")


def load_samples(split: str, root: Path = DATA_ROOT) -> list[dict]:
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=None)
def load_catalog(split: str, root: Path = DATA_ROOT) -> dict:
    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


def catalog_tools(split: str, root: Path = DATA_ROOT) -> dict[str, dict]:
    """{tool name: tool spec} for every tool of the split."""
    return {t["name"]: t for a in load_catalog(split, root)["apps"] for t in a["tools"]}


def fuzzy_params(split: str, root: Path = DATA_ROOT) -> dict[str, set[str]]:
    """{tool name: names of its x-match: fuzzy parameters} (fuzzy only occurs on top-level string parameters)."""
    return {name: {k for k, p in (t["parameters"].get("properties") or {}).items() if p.get("x-match") == "fuzzy"}
            for name, t in catalog_tools(split, root).items()}


def audio_path(split: str, rel: str, root: Path = DATA_ROOT) -> Path:
    return root / split / rel


_ZAIC = {}


def zaic(split: str, root: Path = DATA_ROOT):
    """The zaic_tools package of `split`. Each split ships its own copy, so the module is imported in isolation."""
    if split not in _ZAIC:
        for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
            del sys.modules[name]
        sys.path.insert(0, str(root / split))
        try:
            _ZAIC[split] = importlib.import_module("zaic_tools")
        finally:
            sys.path.pop(0)
        for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
            del sys.modules[name]
    return _ZAIC[split]


def tool_env(split: str, sample_id: str, root: Path = DATA_ROOT):
    """A fresh ToolEnv for one sample (one per sample at inference time - see the competition rules)."""
    return zaic(split, root).ToolEnv(split, sample_id, root=root)
