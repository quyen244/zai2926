# zai2926

Zalo AI Challenge 2026 - LifeMate: a voice agent that turns spoken Vietnamese requests into tool calls.

The competition data is not in this repository (its licence forbids redistribution).
Unzip the data package into `data/train+public_test/` to run anything here.
Per-sample derived files (transcripts, gold hashes, ASR output) are written to `work/`, which is git-ignored.

## Layout

| Path | Content |
|---|---|
| `lifemate/data.py` | paths, loaders, per-split `zaic_tools` import, `tool_env()` |
| `lifemate/scoring.py` | local re-implementation of the official metric (hash gate + call F1) |
| `lifemate/gold.py` | gold replay for train, submission read/write/validate |
| `lifemate/splits.py` | `cv5` stratified folds and the `leave_app_out` split |
| `lifemate/taxonomy.py` | structural and linguistic situation tags |
| `scripts/stage0_harness.py` | builds gold, splits, runs the validator checks -> `reports/stage0.md` |
| `scripts/stage1_labels.py` | label, schema, environment and history analysis -> `reports/stage1_labels.md` |
| `scripts/stage1_taxonomy.py` | tags every train sample -> `reports/stage1_taxonomy.md` |
| `scripts/asr_bench.py`, `scripts/asr_report.py` | ASR benchmark -> `reports/stage1_asr.md` |
| `scripts/stage1_listening.py` | local listening sheet -> `work/listening/index.html` |
| `splits/` | split definitions (sample ids only) |
| `reports/` | aggregate findings (no transcripts) |

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # then put the OpenAI key in .env
.venv/Scripts/python -m pytest tests
```

The analysis scripts only need the standard library plus the competition's `zaic_tools`; the ASR scripts need the venv.
