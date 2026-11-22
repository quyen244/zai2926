---
name: kaggle-datagen
description: Generate synthetic training data (paraphrases, labels, judged rewrites) with hosted LLMs through the Kaggle Benchmarks Model Proxy on the free Kaggle "AI Models" quota, then turn the kept outputs into SFT rows. Use when the user asks to create, augment or label training data with an LLM, mentions the Kaggle AI quota / "daily AI models" / model proxy, or wants more SFT rows for LifeMate.
---

# Generating training data with the Kaggle Model Proxy

The reference implementation is `scripts/paraphrase.py` (rewrites of the final user utterance, 2026-11-09).
Copy its structure for any new generator; do not invent a new client.

## Hard rules (check before anything runs)

1. **Train split only.** Send only train-fold samples (`work/sft/*.train.jsonl` ids). Never val, test, public_test or private_test: labelling test data with any model is forbidden by the competition rules.
2. **Open-weight generator.** The text that becomes training data comes from an Apache-2.0 model (`openai/gpt-oss-120b`, `ibm/granite-4.0-h-small`). Closed models (Gemini, Claude, GPT) may only be pass/fail **judges**, never the source of training text.
3. **Flow first, then wait for a go.** Draw the flow artifact (diagram-design skill, Data flow, Noto Serif; save under `artifact/flows/`), show the filled prompt for one real sample, run a 20-sample trial, and only then the full run, each step after the user's explicit go.
4. **Secrets.** The proxy key lives in `work/synth/mp.env` (git-ignored). Never print it, never commit it, never send it to a Kaggle job. Suppress the auth command's output.
5. **Per-sample output stays in `work/`.** The repo is public: commit code, prompts and aggregate numbers only.
6. **Prompts are files.** Put them in `prompts/<name>.{system,user}.md`. Code only reads them. Few-shot examples inside them must be invented, never real sample text.

## Access

- Token (lasts about 2 h): `.venv/Scripts/kaggle.exe benchmarks auth -y --env-file mp.env`, run inside `work/synth/`.
  It writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`, `MODEL_PROXY_EXPIRY_TIME`.
  `token()` in `scripts/paraphrase.py` refreshes it when the file is older than 100 minutes.
- Endpoint: `MODEL_PROXY_URL` + `/openapi/chat/completions` (currently `https://mp-staging.kaggle.net/models/openapi/chat/completions`, a staging host that may change).
  OpenAI-style body, `Authorization: Bearer <key>`, no streaming, no model listing.
- Quota: $10 per day, $100 per month. Check with `.venv/Scripts/kaggle.exe benchmarks quota` before and after a run (it can lag behind real calls).
- Cost per call is in the response: `usage.cost.input_tokens_cost_nanodollars` + `output_tokens_cost_nanodollars`.

Models that answered (2026-11-09): `openai/gpt-oss-120b`, `ibm/granite-4.0-h-small`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`, `openai/gpt-5.4-nano-2026-03-17`, `anthropic/claude-sonnet-5@default`.
404: `qwen/qwen3-next-80b-a3b-instruct`, `deepseek-ai/deepseek-r1-0528`.

## Settings that matter (measured)

- **`reasoning_effort: "low"` for gpt-oss-120b.** With the default it spent about 790 tokens per short rewrite and the proxy queued calls (median 816 s per call). With `low`: median 2.1 s.
- **16 workers** in a thread pool; retry 429 / 5xx with exponential backoff (see `ask()`).
- **`max_tokens` 4000**, and treat a missing `message` or empty `content` as an error row, not a crash.
- **Parse defensively.** Find the first JSON list in the reply. Judges sometimes answer one JSON object per line: fall back to line-by-line (`parse_objs()`).

## Recipe

1. **Generate.** One call per sample asks for k outputs (k = 2) as a JSON list. Pass the phrases that must survive verbatim ("keep" list).
2. **Rule checks (free, local).** Reject a candidate that loses a keep phrase, changes the numbers or resolved dates after `lifemate.normalize`, loses the condition word of a conditional sample, duplicates the original or a sibling, or is outside 0.5x to 2x the length.
3. **Judges (paid).** Send the survivors with the gold calls to every judge; keep a candidate only if all judges return `{"ok": true}`. A single gpt-oss judge passed every known meaning change; Gemini + Claude together caught all of them.
4. **Second attempt** when fewer than k survive.
5. **Cache everything.** One JSONL line per call in `work/synth/<name>.jsonl`, keyed by (sample id, attempt, kind = `gen` | `judge:<model>`) plus a hash of the prompt files and model. A rerun calls only what is missing; changing a prompt invalidates only its own calls.
6. **Export the kept list** (`--show` makes no calls and writes `work/synth/<name>.kept.jsonl`).
7. **Build SFT rows** from the kept list: `scripts/plan_data.py --para work/synth/paraphrase.kept.jsonl --out work/sft_para` (never overwrite `work/sft`). Then `scripts/sft_filter_data.py` for a top-5-app tool list.
8. **Verify the rows**: replay failures 0, no row over 10240 tokens, original rows byte-identical to the base build, and print one sample's original row next to its new row.

Commands:

```bash
.venv/Scripts/python.exe -X utf8 scripts/paraphrase.py --trial 20
```
```bash
.venv/Scripts/python.exe -X utf8 scripts/paraphrase.py --all > work/synth_all.txt 2>&1
```
```bash
.venv/Scripts/python.exe -X utf8 scripts/paraphrase.py --show
```

## Reference numbers (paraphrase run, 2026-11-09)

| | Value |
|---|---|
| Samples | 1176 train-fold |
| Kept | 2115 / 2352 rewrites (1015 samples with 2) |
| Calls | 4625, median 2.1 s |
| Wall time | 100 min with 16 workers |
| Cost | $5.46 (estimate before the run was $4.6) |
| Main rejects | Gemini 509, Claude 217, dates 176, numbers 154, condition word 84 |

Some number and date rejects are false: the normaliser turns "tầng 2 hết" into a time and "cả hai sáu giờ" into "26 giờ".

## Report back

Report each run as before / after blocks: samples, kept outputs, reject reasons, cost from the quota command, wall time, and one sample shown side by side in the `show_gt` layout (`    -> name {args}` / `         = {...}`).
