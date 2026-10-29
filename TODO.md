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

## Stage 1 - Understand the data (mostly done)

Reports: `reports/stage1_summary.md` (start here), `stage1_labels.md`, `stage1_taxonomy.md`, `stage1_asr.md`.

- [~] 1.1 Listening pass: sheet built (`python scripts/stage1_listening.py` -> `work/listening/index.html`, 170 clips). Needs a human to listen and export `listening.csv`.
  - [ ] Check whether reference transcripts are noisy (a suspicious one was seen).
- [x] 1.2 ASR benchmark on 400 stratified train clips: PhoWhisper-large best (WER 0.246, gold-arg recall 0.570, hallucination 1.0%) vs Whisper large-v3 (0.297 / 0.492 / 5.8%).
  - [~] Whole-utterance hallucination: not explained by level; longer leading silence. Peak-norm or VAD fixes 11/25 (large-v3). Probe PhoWhisper's 4 cases.
- [x] 1.3 Label anatomy: gold read-call conventions, ID provenance (25% from history), ask patterns, empty-gold subtypes.
- [~] 1.4 Taxonomy: regex tags for all train (`work/tags.jsonl`); LLM tagger ready (`scripts/llm_tag.py`, `taxonomy/taxonomy.json`).
  - [ ] Write the prompts in `prompts/taxonomy_tagging.{system,user}.md` (team).
  - [ ] Run 40 samples, check quality, then all train.
  - [ ] Hand-check `work/tags_review.csv` (200 rows) for regex precision.
- [x] 1.5 Schema and env: shared app schemas identical train vs public; 11 public-only apps; silent failures return `ok: true, done: false`.
- [x] 1.6 History: 66% of samples; every earlier public user turn is audio only (1299 turns) -> ASR on history too.

## Stage 2 - Data processing (next)

- [~] 2.1 ASR pipeline (`scripts/asr_run.py`): public_test 1933/1933 done, train in progress (2100/5036, resumable). Output `work/asr/phowhisper/<split>.jsonl`.
- [ ] 2.1 Hallucination guard (detect filler phrase / low logprob -> re-decode with VAD / normalisation).
- [x] 2.2 Text normaliser (`lifemate/normalize.py`: numbers/money/clock time/relative dates), 26 tests. Checked on train gold args: ISO dates found in 423/603 (the rest mostly default to the current date, left to the prompt builder); money/time misses are mostly amounts computed from app data or part-of-day inferred from "tối nào".
- [ ] 2.3 `build_prompt(sample)`: installed-tool schemas, compact history with IDs, context.
- [ ] 2.4 SFT trajectories from train (reference + ASR text variants), replay-verified.

## Stage 3 - Baselines

- [ ] B0 reference transcript + LLM tool loop.
- [ ] B1 ASR transcript.
- [ ] B2 + validate-before-call, `done` check, step cap, few-shot retrieval.
- [ ] Per-slice report (taxonomy, seen vs unseen apps); first public leaderboard submission.

## Stage 4 - Upgrades I

- [ ] ASR fine-tune (LoRA) on ~5000 train utterances; contextual biasing; N-best rescoring.
- [ ] Agent SFT (LoRA) with schema shuffling; synthetic data for unseen apps verified in env.
- [ ] Act / ask / empty decision tuned on the official score.

## Stage 5 - Upgrades II (only if time)

- [ ] Rejection sampling -> DPO -> GRPO with the official score as reward.

## Stage 6 - Docker (hard deadline 2026-11-16)

- [ ] Offline image with weights, deterministic decoding, full public_test rehearsal, timing/VRAM, checksum, Drive link.
- [ ] Ask organisers: compute limits, external API policy.

## Stage 7 - Private test (2026-11-20 to 11-23)

- [ ] Run the frozen Docker, validate, submit.
