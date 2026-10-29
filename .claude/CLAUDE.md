# Project memory - LifeMate (Zalo AI Challenge 2026)

Read these first when resuming, in this order:

1. `TODO.md` - progress ledger; update it whenever a task is done (tick + one-line evidence).
2. `reports/stage1_summary.md` - what is known about the data so far.
3. `docs/lifemate-brief.html` - problem, scoring, difficulties, tech stack.
4. `docs/lifemate-roadmap.html` - phases with exit gates and dates.
5. `README.md` - code layout and setup.

## Machines (how the user names them)

- "server rtx": the RTX 5060 Ti GPU server work session.
- "local": the user's current workstation (this Windows machine).
- "kaggle server": work done on Kaggle.

## Competition constraints (official regulations, received 2026-10-30)

These bind the FINAL solution (the Docker image scored on private_test). They do not apply to dev-only
tools like `scripts/llm_tag.py` (explicitly allowed: "synthetic data... during development", train only,
never used at inference) or to a local harness smoke-test that is never uploaded to the portal.

- **Model size**: whole system (ASR + LLM + anything else, e.g. a retrieval embedder) < 6B params total.
  The LLM itself < 5B params, context <= 16K (16,384) tokens. PhoWhisper-large is ~1.55B, so the LLM budget
  is roughly <4B if every other component (ASR, any embedder) is to fit under the 6B ceiling too.
  This replaces the earlier "Qwen3 8B to 32B" idea in the brief - neither size is legal. Target a Qwen3
  tier under 5B (e.g. ~4B), and double-check context budget: Stage 2.3 measured ~5.5k-16k tokens per
  prompt on real samples (tool schemas + history), some samples may need trimming to stay under 16K.
- **Offline, no external API/service at inference.** Scoring hardware: 1x NVIDIA A30 (24GB), 16-core Xeon
  Gold 6442Y, 64GB RAM, provided by the organisers (not server rtx, which is only for our own dev/fine-tune).
  30s per sample timeout; a sample that times out or errors scores 0.
- **Reproducibility**: fixed seeds, deterministic decoding (greedy / temperature 0) for both ASR and the
  LLM. If a re-run differs, the organisers average 5 runs.
- Public data and open-source models released before 2026-10-26 only, licence permitting.
- Labelling public_test or private_test (by hand or with any external service/model) to train or tune the
  solution is NOT allowed. The taxonomy LLM tagger must only ever run on `train` for this reason, never on
  public_test/private_test, regardless of what `--asr` makes technically possible.
- Exactly one ToolEnv per sample; every call made on it must appear in the submission (already followed:
  `lifemate/agent.py`, `lifemate/gold.py`).
- **Round 1 (public_test, now to 2026-11-16) vs the final Docker**: round 1 scores are "for reference only
  during development" and the regs don't explicitly scope "offline, no external API" to round 1 - but they
  don't explicitly exempt it either. Current status: unresolved. Until clarified with organisers, treat any
  OpenAI-API-based agent run (`scripts/run_baseline.py`'s current `gpt-6-luna` client) as internal harness
  validation only - do not upload its submission.jsonl to the portal. The real Stage 3/4 target is a local
  sub-5B model; once chosen, re-run baselines on that and those results are safe to submit regardless of
  how round 1 is eventually interpreted.

## Rules for this repo

- The repo `quyen244/zai2926` is PUBLIC. Never commit competition data or anything per-sample derived from it (transcripts, labels, hashes, ASR output): those go in `work/` (git-ignored). Reports hold aggregates only.
- Secrets live in `.env` (git-ignored): `OPENAI_API_KEY`, optional `LLM_MODEL` (default `gpt-6-luna`; stay on the luna tier or cheaper), `LLM_RPM`, `LLM_TPM`.
- LLM prompts are written by the user in `prompts/`; code only reads them. Taxonomy definitions: `taxonomy/taxonomy.json`.
- Never read `data/**/<split>_environment/` directly; use `lifemate.data.tool_env()` (competition rule).
- Commits: no co-author line.
- Running scripts: always smoke-test first (small `--n`), then optimise runtime using real measured numbers (batch size, VAD, compute type, caching). Stay at "just right": one or two measured fixes, no over-engineering or endless tuning.

## Environment to rebuild on a new server

- Data: unzip `train+public_test.zip` to `data/train+public_test/`.
- `python -m venv .venv` (Python 3.12) and install as in `README.md`; on Linux skip `lifemate/cuda_dlls.py` concerns (it is a no-op without the Windows pip CUDA wheels).
- Regenerate `work/`: `python scripts/stage0_harness.py`, `python scripts/stage1_taxonomy.py`, then ASR: `python scripts/asr_bench.py --models large-v3 large-v3-turbo <phowhisper ct2 dir> --n 400`.
- PhoWhisper CT2 conversion: `ct2-transformers-converter --model vinai/PhoWhisper-large --output_dir work/models/phowhisper-large-ct2 --quantization float16 --copy_files preprocessor_config.json tokenizer_config.json vocab.json merges.txt normalizer.json special_tokens_map.json added_tokens.json` (needs `transformers` + `torch`).
