# Project memory - LifeMate (Zalo AI Challenge 2026)

Read these first when resuming, in this order:

1. `TODO.md` - progress ledger; update it whenever a task is done (tick + one-line evidence).
2. `reports/stage1_summary.md` - what is known about the data so far.
3. `docs/lifemate-brief.html` - problem, scoring, difficulties, tech stack.
4. `docs/lifemate-roadmap.html` - phases with exit gates and dates.
5. `README.md` - code layout and setup.

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
