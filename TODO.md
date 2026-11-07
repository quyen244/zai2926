# TODO - LifeMate (ZAC 2026)

Ledger of the roadmap in `docs/lifemate-roadmap.html`.
Context and scoring: `docs/lifemate-brief.html`.
Deadlines: Docker solution 2026-11-16 23:59 (GMT+7); private test 2026-11-20 09:00 to 2026-11-23 23:59.
Legend: `[x]` done and verified, `[~]` partly done, `[ ]` not started.

## Stage 0 - Scoring harness (done 2026-10-28)

- [x] Local scorer = official formula (`lifemate/scoring.py`), 19 tests pass (`python -m pytest tests`).
- [x] Gold replay of train: 1545/1545 reproduce `final_state` and `tool_results`; gold self-score 100.00.
- [x] Floor (submit `[]` everywhere): 7.18 on train.
- [x] Splits: `splits/cv5.json`, `splits/leave_app_out.json` (9 held-out train-only apps, 241 val samples).
- [x] Submission validator (`lifemate/gold.py: validate_submission`).
- [ ] Confirm the scorer assumptions (empty arg = absent, token-F1 tokenisation) against the first public leaderboard score.
  - [ ] **Extra optional arguments** (pred has every gold arg plus more, e.g. train_0007 `label`, `repeat`): `lifemate/scoring.py` requires the same argument set, so the call does not match. The rules only say "arg thường giống hệt, arg fuzzy token-F1 > 0.6". On the 2026-11-05 Qwen run this choice alone moves B0 5.33 -> 17.17 and B1 2.67 -> 6.83. Ask the organisers or settle it with the leaderboard; until then keep strict and make the prompt forbid unrequested optional args.

## Stage 1 - Understand the data (mostly done)

Reports: `reports/stage1_summary.md` (start here), `stage1_labels.md`, `stage1_taxonomy.md`, `stage1_asr.md`.

- [~] 1.1 Listening pass: sheet built (`python scripts/stage1_listening.py` -> `work/listening/index.html`, 170 clips). Needs a human to listen and export `listening.csv`.
  - [ ] Check whether reference transcripts are noisy (a suspicious one was seen).
- [x] 1.2 ASR benchmark on 400 stratified train clips: PhoWhisper-large best (WER 0.246, gold-arg recall 0.570, hallucination 1.0%) vs Whisper large-v3 (0.297 / 0.492 / 5.8%).
  - [x] Whole-utterance hallucination: root cause per quyen (listened to the clips) is simply that those clips are hard to hear/understand, not a pipeline artifact. On PhoWhisper (the model used from Stage 2 on) it is 1.0% (4/400); accepted, no fix planned.
- [x] 1.3 Label anatomy: gold read-call conventions, ID provenance (25% from history), ask patterns, empty-gold subtypes.
- [~] 1.4 Taxonomy: regex tags for all train (`work/tags.jsonl`); LLM tagger ready (`scripts/llm_tag.py`, `taxonomy/taxonomy.json`).
  - [x] Write the prompts in `prompts/taxonomy_tagging.{system,user}.md` (quyen), with 2 invented few-shot examples (medium + hard, incl. a prompt-injection-resistance case). Dry-run renders cleanly, no placeholder errors.
  - [ ] Run 40 samples, check quality, then all train.
  - [ ] Hand-check `work/tags_review.csv` (200 rows) for regex precision.
- [x] 1.5 Schema and env: shared app schemas identical train vs public; 11 public-only apps; silent failures return `ok: true, done: false`.
- [x] 1.6 History: 66% of samples; every earlier public user turn is audio only (1299 turns) -> ASR on history too.

## Stage 2 - Data processing (next)

- [x] 2.1 ASR pipeline (`scripts/asr_run.py`): public_test 1933/1933 and train 5036/5036 clips (final + history user turns), no empty hypothesis. Output `work/asr/phowhisper/<split>.jsonl`.
- [x] 2.1 Hallucination guard (flag -> re-decode with peak-norm + VAD): fired on 4 train clips and 0 public_test clips; 0 empty, 0 with compression_ratio > 2.4 after the guard.
- [x] 2.2 Text normaliser (`lifemate/normalize.py`: numbers/money/clock time/relative dates), 26 tests. Checked on train gold args: ISO dates found in 423/603 (the rest mostly default to the current date, left to the prompt builder); money/time misses are mostly amounts computed from app data or part-of-day inferred from "tối nào".
  - [x] 2026-11-04: lowercase ASR fix. PhoWhisper output has no capitals, so "anh mười" became "anh 10" and "cô mai" a tomorrow hint. Now a lone number word after a kin word stays unless a unit follows, and "mai" after "cô"/"cô giáo" (after "chị" unless "có" or a number follows) is a name. On all 5036 train user turns: false tomorrow hints on ASR 46 -> 23, kin-number names kept 58/64 -> 61/64, real tomorrow hints kept 514 -> 514, reference-text hints unchanged. 7 new tests.
- [x] 2.3 `build_prompt(sample)` (`lifemate/prompt.py`): system rules + context, tools of installed apps only, history replayed as real tool-calling turns. `scripts/prompt_mock.py` prints a labelled mock (SYSTEM/TOOLS/MESSAGES) for review; 4 structural tests pass. Median prompt ~16k chars / ~5.5k tokens on a 4-history-turn train sample.
  - [x] 2026-11-04: the system rules told the model to ask "bằng lời" (in words), but asking is scored only as a `sys.ask_missing_info` call (217/1545 gold use it; dropping that call on those samples scores 54.3 on average, 27.6% of them 0). Rule now names the tool, its arguments, and "do the parts that are complete first, then ask and stop" (as the gold does). Test added; effect on a real model not measured yet.
- [ ] 2.4 SFT trajectories from train (reference + ASR text variants), replay-verified.

## Stage 3 - Baselines

- [~] Agent loop built (`lifemate/agent.py` + `scripts/run_baseline.py`): one ToolEnv per sample, LLM tool-calling via `lifemate/llm.py: chat_tools`, writes a submission.jsonl matching `data/.../sample_submission.jsonl` exactly (verified byte-for-byte on keys). Smoke-tested end to end on 3 train samples (B0, using `gpt-6-luna` via the OpenAI API): scores 0.36 / 1.0 / 0.0, harness confirmed working.
  - Fixed along the way: `build_tools()` was dropping `sys.ask_missing_info` (always available per the catalog, not gated by installed_apps). OpenAI's function name pattern rejects ".", so `chat_tools` maps "app.tool" <-> "app__tool" on the wire only. `gpt-6-luna` needs `reasoning_effort: "none"` to use tools on `/v1/chat/completions`.
  - **Compliance note (2026-10-30 regs)**: the official LLM must be < 5B params, fully offline, no external API at inference. The current `gpt-6-luna` runs are internal harness validation only - not submitted to the portal, not the final architecture. See `.claude/CLAUDE.md` "Competition constraints".
- [x] Model locked 2026-10-30: **PhoWhisper-large (~1.55B) + Qwen3-4B-Instruct (~4B)**, ~5.5B total, under the 6B cap. See the "ASR & Agent Training Plan" artifact for architecture + fine-tune data design.
- [ ] Serve Qwen3-4B locally on server rtx (vLLM, OpenAI-compatible tool-calling endpoint); verify context fits <=16K tokens on real prompts (Stage 2.3 measured up to ~16k chars / ~5.5k tokens on a 4-turn sample - check the longer-history tail too).
- [~] Re-run B0/B1 on Qwen3-4B. 2026-11-05, Kaggle T4 x2 (`kaggle/baseline/`, Qwen3-4B-AWQ, 2 shards, 36 min), first 100 train samples: **B0 5.33, B1 2.67** (hash ok 25 / 16), 0 failures. Main errors: invented optional args (34 calls had all gold args plus extras), stopping after the first read (175 predicted vs 275 gold calls), extra sys.ask_missing_info (26), phone leading zero lost. Results cached in `work/kaggle/cache/`. Full train not run yet.
  - 2026-11-07 QVikhr-3-4B-Instruction (Qwen3-4B fine-tune for RU/EN, FP16, `kaggle/vikhr/`, same code a209ba03 and samples): **B0 7.17, B1 3.00** (hash ok 29 / 18), 0 failures, 93 min run (2.6x slower than AWQ, KV cache 28k tokens/GPU). Paired vs Qwen: B0 +1.83 (95% bootstrap CI -0.83..+4.50, better on 5 samples, worse on 1), B1 +0.33 (CI -1.67..+2.33). Not a significant gain; more extra optional args (1a 66% vs 51%). See `reports/error_eda.md`.
- [ ] B2 + validate-before-call, `done` check, step cap, few-shot retrieval (retrieval must not add real params if it would blow the 6B budget - no embedding model, or a tiny one counted in).
  - 2026-11-07 tool-list filtering measured offline on 1374 train samples, ASR text, name + description context (`reports/retrieval_dense.md`, `reports/retrieval_bm25.md`): BGE-M3 best app@5 81.4% sample recall (19.5 tools kept) vs BM25 78.1%; tool@20 75.0% vs 69.9%. No filter is near the ~100% recall it needs (a dropped tool = 0); BGE-M3 (~568M) would also push the system to ~6.1B. RRF fusion of dense + BM25 app rankings: app doc @5 85.3%, fusing all four app lists (dense/BM25 x max/app) @5 86.2% (19.9 tools); +4.8 over dense alone, still far from safe. tev1:0.8b group classifier: recall 43.8% on 40 samples. Keep all installed tools for now.
- [ ] Per-slice report (taxonomy, seen vs unseen apps); first public leaderboard submission (user uploads via the portal, 5/day quota) - only once using a compliant (or explicitly organiser-cleared) model.

## Stage 4 - Upgrades I

- [ ] ASR fine-tune (LoRA) on ~5000 train utterances; contextual biasing; N-best rescoring.
  - [x] 2026-11-02: decode-time hotwords tested and rejected. 60 English/brand terms from cv folds 1-4, tested on fold 0 (127 clips with such terms + 80 without): terms kept 78/180 -> 86/180, but WER 0.279 -> 0.306 on those clips and 0.214 -> 0.296 on the controls; `initial_prompt` gives identical output. A very short list (SMS, Zalo, SJC) is untested.
- [ ] Agent SFT (LoRA) with schema shuffling; synthetic data for unseen apps verified in env.
- [ ] Act / ask / empty decision tuned on the official score.

## Stage 5 - Upgrades II (only if time)

- [ ] Rejection sampling -> DPO -> GRPO with the official score as reward.

## Stage 6 - Docker (hard deadline 2026-11-16)

- [x] Ask organisers: compute limits, external API policy. **Answered 2026-10-30** (full regs in `.claude/CLAUDE.md`): system <6B params (LLM <5B, context <=16K), scoring on 1x A30/24GB + 16-core Xeon + 64GB RAM, 30s/sample timeout, fully offline/no external API at inference, deterministic decoding, models/data from before 2026-10-26.
  - [ ] Still open: whether "offline, no external API" also binds round-1 (public_test) submissions, or only the final Docker. Ask directly; until answered, don't submit an OpenAI-API-based run to the portal.
- [ ] Offline image with weights, deterministic decoding, full public_test rehearsal, timing/VRAM, checksum, Drive link.

## Stage 7 - Private test (2026-11-20 to 11-23)

- [ ] Run the frozen Docker, validate, submit.
