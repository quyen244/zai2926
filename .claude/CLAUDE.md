# Project memory - LifeMate (Zalo AI Challenge 2026)

Read these first when resuming, in this order:

1. `TODO.md` - progress ledger; update it whenever a task is done (tick + one-line evidence).
2. `reports/stage1_summary.md` - what is known about the data so far.
3. `docs/lifemate-brief.html` - problem, scoring, difficulties, tech stack.
4. `docs/lifemate-roadmap.html` - phases with exit gates and dates.
5. `README.md` - code layout and setup.

## Machines (how the user names them)

- "server" (plain, from 2026-11-04 on): the Kaggle server. GPU jobs run there as Kaggle kernels pushed with the
  `kaggle` CLI (credential already in `C:/Users/Admin/.kaggle/`, never print or copy it); do not run models on "local".
  Job sources live in `kaggle/<job>/` (script + `kernel-metadata.json`, private kernels).
- "server rtx": the RTX 5060 Ti GPU server work session (only when the user says "rtx").
- "local": the user's current workstation (this Windows machine).
- "kaggle server": same as "server".

## How the user wants runs handled (2026-11-05)

- **Never launch a run on your own.** Before running anything (a Kaggle job, an upload, a long script), draw the flow
  first (small hand-drawn HTML artifact: input -> each step -> output, what runs where, what is cached) and wait
  for the user's explicit go. Preparing files locally is fine; pushing/uploading/executing is not, until approved.
- **Run, don't benchmark.** The latency question is closed for now (numbers below). Do not add timing probes or
  re-measure engines unless the user asks; spend GPU time on producing results (baselines, scores).
- **Two GPUs = data sharding (data parallelism).** Kaggle "T4 x2": one model copy per GPU, samples split in half
  (GPU 0 gets one shard, GPU 1 the other), both start at once and run independently, results merged and scored
  at the end. Not one job per GPU, not tensor parallel (a 4B AWQ model fits one T4).
- **Cache results so nothing is recomputed.** Each finished sample is written to a per-sample cache keyed by
  (model, variant, sample id) and fed back to the next job as input, so a larger run (e.g. all train after a
  100-sample trial) only runs the samples not yet done.
- **Staged runs:** small trial first (100 train samples), check it, then the full split.
- Data may be uploaded to a **private** Kaggle dataset when a job needs it (approved 2026-11-05).
- **Every experiment result is reported with the error EDA table (2026-11-07).** After any run (baseline, prompt
  change, new model), merge its cache, run `scripts/error_eda.py` and show the user its table (% of the run's
  samples per group: 1 wrong args, 1a extra / 1b missing / 1c wrong value, 2 under, 3 over, 4 wrong tool,
  5 no-tool, 6 lookup skip, 7 multi-step, hash) next to the score, and against the previous run when there is one.
  Group definitions live in `lifemate/eda.py`. Drill-down: `scripts/error_ids.py` (ids per group, default 1a 1b 2 3)
  and `scripts/compare_sample.py <id> --text gold|asr` (gold vs predicted side by side with the reason).

## Kaggle GPU jobs: fixed recipe (decided 2026-11-04, do not re-litigate)

- CLI: `.venv/Scripts/kaggle.exe` (not on PATH). Push `kaggle kernels push -p kaggle/<job> --accelerator NvidiaTeslaT4`,
  poll `kernels status <id>`, fetch `kernels output <id> -p work/kaggle/<job>`.
- On Windows, run the CLI from inside the folder with `-p .`: a relative `-p work/x` breaks the upload temp path.
- Data for jobs: private dataset `quyenuit24/lifemate-data` (`scripts/kaggle_pack.py` builds it, then
  `datasets version -p . -m msg`); baseline job `kaggle/baseline/` (2 shards, result cache, merge with
  `scripts/kaggle_merge_cache.py`).
- GPU: push with `--accelerator NvidiaTeslaT4`, which is "GPU T4 x2" (two T4 visible, seen in nvidia-smi). Asking for `NvidiaL4` silently falls back to a T4 (seen twice).
- The T4 is compute capability 7.5: no FlashAttention 2. Always use vLLM with
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`, set in `os.environ` before `import vllm`, unconditionally.
  Do not use the default (FlexAttention: 12 prompts batched 98 s vs 22 s with Triton) and do not use HF transformers
  (SDPA math path, OOM as soon as prompts are batched).
- Model: `Qwen/Qwen3-4B-AWQ` (pre-quantised; loads ~100 s faster than bitsandbytes and is faster at step 1).
- Pins: `vllm==0.10.2` then `transformers==4.57.1` (vLLM 0.10.2 breaks on transformers 5.x tokenizer API).
- Load: `LLM(model=..., max_model_len=16384, gpu_memory_utilization=0.88, dtype="half")`; Qwen3 chat template with
  `enable_thinking=False`. Loading costs ~136 s (torch.compile + CUDA graphs).
- Measured on T4 (kaggle/vllm_awq v1, 2026-11-05): one agent step ~22 s = prefill of a ~4.8k-token prompt ~11 s
  (~440 tok/s) + decode ~165 ms per token, independent of context length and of batch size. No warm-up cost
  (fresh prompts as slow as repeated ones). Prefix cache works (cached prompt: ~5 s) but 12 x 4.8k prompts overflow it.
  Batched runs after earlier passes look fast only because of prefix-cache hits; do not read them as per-request cost.
- T4 latency is not the A30's: the scoring A30 (sm80) has FlashAttention 2; use T4 numbers for relative comparisons only.
- Template job: `kaggle/vllm_awq/` (results of earlier benches in `work/kaggle/`).

## Competition constraints (official regulations, received 2026-10-30)

These bind the FINAL solution (the Docker image scored on private_test). They do not apply to dev-only
tools like `scripts/llm_tag.py` (explicitly allowed: "synthetic data... during development", train only,
never used at inference) or to a local harness smoke-test that is never uploaded to the portal.

- **Model size**: whole system (ASR + LLM + anything else, e.g. a retrieval embedder) < 6B params total.
  The LLM itself < 5B params, context <= 16K (16,384) tokens. **Locked choice (2026-10-30): PhoWhisper-large
  (~1.55B) + Qwen3-4B-Instruct (~4B), total ~5.5B**, leaving ~0.45B headroom for Stage 2 extras (B2
  retrieval must not add a real embedding model on top, or it has to come out of that headroom).
  Context budget: Stage 2.3 measured ~5.5k-16k tokens per prompt on real samples (tool schemas + history);
  Qwen3-4B natively supports well beyond 16K, so the 16K cap is a rules limit, not a model limit - some
  samples may still need trimming (truncate older history / tool descriptions) to stay under it.
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

- **Human-friendly, easy-on-the-eyes output (2026-11-07).** Everything a script prints or writes for the user to read
  (traces, sample views, submission lines, gold vs predicted) is laid out for reading and comparing, never as a raw
  one-line JSON dump: one tool call per line as `    -> name {args}`, its result on the next line as `         = {...}`
  (the `scripts/show_gt.py` layout), a short header line per sample, gold and predicted in the same layout so they
  line up. Raw JSON only behind a flag (e.g. `trace_sample.py --request`).
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
