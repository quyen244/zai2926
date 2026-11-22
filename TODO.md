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
- [x] 2026-11-22 PhoWhisper P2 fold0 audit and Kaggle decode sweep: 1,009 clips, 14.3624% WER in the prior output; same-adapter beam-5 decode without the one harmful VAD retry 14.2867%, beam 8 14.3693%, train-only name prompt 15.3465%. Keep beam 5 without a prompt. Evidence: `work/asr_p2_audit.md` and `work/kaggle/asr_p2_decode_sweep_tquyen/out/metrics.json`.
- [x] 2.2 Text normaliser (`lifemate/normalize.py`: numbers/money/clock time/relative dates), 26 tests. Checked on train gold args: ISO dates found in 423/603 (the rest mostly default to the current date, left to the prompt builder); money/time misses are mostly amounts computed from app data or part-of-day inferred from "tối nào".
  - [x] 2026-11-04: lowercase ASR fix. PhoWhisper output has no capitals, so "anh mười" became "anh 10" and "cô mai" a tomorrow hint. Now a lone number word after a kin word stays unless a unit follows, and "mai" after "cô"/"cô giáo" (after "chị" unless "có" or a number follows) is a name. On all 5036 train user turns: false tomorrow hints on ASR 46 -> 23, kin-number names kept 58/64 -> 61/64, real tomorrow hints kept 514 -> 514, reference-text hints unchanged. 7 new tests.
- [x] 2.3 `build_prompt(sample)` (`lifemate/prompt.py`): system rules + context, tools of installed apps only, history replayed as real tool-calling turns. `scripts/prompt_mock.py` prints a labelled mock (SYSTEM/TOOLS/MESSAGES) for review; 4 structural tests pass. Median prompt ~16k chars / ~5.5k tokens on a 4-history-turn train sample.
  - [x] 2026-11-04: the system rules told the model to ask "bằng lời" (in words), but asking is scored only as a `sys.ask_missing_info` call (217/1545 gold use it; dropping that call on those samples scores 54.3 on average, 27.6% of them 0). Rule now names the tool, its arguments, and "do the parts that are complete first, then ask and stop" (as the gold does). Test added; effect on a real model not measured yet.
- [x] 2.4 SFT trajectories from train (reference + ASR text variants), replay-verified. `scripts/plan_data.py` -> work/sft/{plan,traj}.{train,val,test}.jsonl: 2352 train rows (1176 samples x gold/asr), val 60, test 309 (cv5 fold 0), 0 replay failures, 0 rows over 10240 tokens.
  - [x] 2026-11-09 plan v2: after the first step a literal is written only if visible when the plan is written, else "?". Fix the same day: "visible" is checked on the normalised text the prompt shows (a number said in words was "?" though the model sees digits): 96/2352 train rows and 14/309 test rows changed, only "?" -> value; traj rows byte-identical.
  - [x] 2026-11-09 paraphrase data (`scripts/paraphrase.py`, gpt-oss-120b via the Kaggle Model Proxy, Gemini + Claude judges): 2115/2352 rewrites kept (1015 samples with 2, 85 with 1), $5.46, 100 min. `plan_data.py --para work/synth/paraphrase.kept.jsonl --out work/sft_para`: 4467 train rows per dataset, val/test identical to work/sft. Not trained yet (~16 h at the A v2 rate, over the 12 h session).

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
  - 2026-11-07 tool filter tried on the model (`kaggle/app_filter/`, Qwen3-4B-AWQ, same 100 samples, only change `run_sample(keep_apps=...)` = RRF all4 top-5 apps + contacts + sys, 41.7 -> 20.3 tools): **B0 4.07 (was 5.33), B1 2.80 (was 2.67)**, paired diff B0 -1.3 points (95% CI -4..+1), B1 +0.1 (CI -2..+2); only 2-3 samples change score, none of the losses is a filtered-out gold app. Run time 36 -> 17 min (mean 132 -> 64 s per sample). Error mix unchanged (over-calls, extra args, skipped lookups). Verdict: fewer tools does not raise the score; it only halves the run time.
- [ ] Per-slice report (taxonomy, seen vs unseen apps); first public leaderboard submission (user uploads via the portal, 5/day quota) - only once using a compliant (or explicitly organiser-cleared) model.

## Stage 4 - Upgrades I

- [ ] ASR fine-tune (LoRA) on ~5000 train utterances; contextual biasing; N-best rescoring.
  - [~] 2026-11-15 PhoWhisper-large LoRA (r8 a16 q,v, lr 1e-4, 1 epoch over 39172 clips: real + TTS batch 1 + 2, folds 1-4 only), 2 Kaggle jobs on quyenuit24, 2449 steps, resume kept the lr schedule. Val loss on 300 fold-0 real clips 0.265 (half) -> 0.239 (end). Adapter in `work/kaggle/phowhisper_q_p2/out/adapter`.
  - [x] 2026-11-15 decode (`kaggle/asr_decode`, merged LoRA -> CT2 float16, same decoding as asr_run.py): fold 0 (held out) WER 0.2596 -> **0.1436**, gold-arg recall 0.602 -> 0.724 (98 strings), hallucination 0.97% -> 0%; folds 1-4 WER 0.2573 -> 0.1329. T4 re-decode of base fold 0: 0.2586 (matches local). Output style changed: capitals, punctuation, digits (319 hyps with digits vs 11). All 5036 clips in `work/asr/phowhisper_ft/train.jsonl`; CT2 model `work/kaggle/asr_decode/out/phowhisper-ft-ct2` (3.1 GB). Agent score on the new ASR text not measured yet.
  - [x] 2026-11-02: decode-time hotwords tested and rejected. 60 English/brand terms from cv folds 1-4, tested on fold 0 (127 clips with such terms + 80 without): terms kept 78/180 -> 86/180, but WER 0.279 -> 0.306 on those clips and 0.214 -> 0.296 on the controls; `initial_prompt` gives identical output. A very short list (SMS, Zalo, SJC) is untested.
- [ ] Agent SFT (LoRA) with schema shuffling; synthetic data for unseen apps verified in env.
  - [x] 2026-11-09 plan ablation, QLoRA Qwen3-4B on Kaggle T4, test fold 0 (309, gold text, `reports/plan_sft_ablation.md`): base 18.32, A v1 plan+SFT 44.93, A v2 ("?" plans) 46.39, B trajectory SFT 49.32, B + bge-m3/BM25 top-5 app filter 50.39. A v2 - B = -2.9 (95% CI -8..+2). Adapters in the private HF repo.
  - [~] 2026-11-14 item 2, ID guard (`lifemate/agent.py id_guard`): B2 + guard on fold 0 **63.01** (B2 62.13, hash ok 221 vs 218). 300 recovery rows (`scripts/recovery_data.py`: blocked invented-ID call, untrained, then lookup + real call; hash = gold on all 300) + 2115 B2 rows, from B2, 1 epoch, lr 1e-5 (5.5 GPU h, val loss 0.0327 -> 0.0337). **Recovery + guard on fold 0: 63.23** (12 samples up, 12 down; guarded samples 9, mean 56.1 vs 32.5 for B2 + guard; self-correction slice 53.6 vs 52.7). Kept as the current adapter (user, 2026-11-15). Fixed along the way: `lifemate/sft.py render` put loss on the last n assistant messages, not the `train`-flagged ones (93/300 recovery rows would have trained the invented call; 32,634 old rows render identically after the fix).
  - [ ] Item 1, rationale (`scripts/rationale_data.py`, gpt-oss-120b, code-verified): entity / reason lines before tool calls. Added 2026-11-14 for self-correction utterances ("à không", "à nhầm", "à mà thôi", "thôi khỏi"; 130/1545 train, 27 in fold 0; B2 + guard 50.4 on them vs 64.2 on the rest, mostly doing the cancelled action):
    - [ ] a "người dùng đổi ý" line before the first call (the cancelled action absent from the gold calls, the kept one matching them), ~20 tokens, no thinking mode (latency: T4 ~165 ms/token);
    - [ ] synthetic self-correction utterances on folds 1-4 (gpt-oss inserts a wrong value then the fix, gold calls unchanged; code checks the right value is in the text and the replay hash equals gold);
    - [ ] report the self-correction slice score separately (27 samples, ~0.3 points each on the total).
- [ ] Act / ask / empty decision tuned on the official score.

## Stage 5 - Upgrades II (only if time)

- [ ] Rejection sampling -> DPO -> GRPO with the official score as reward.
  - Initial agreed order (2026-11-15) used no app filter; superseded on 2026-11-16 by the selected E5 top-7 filter for inference.
    Each job still needs the user's go.
    - [x] 1. PhoWhisper LoRA, 1 epoch over 2 jobs (fold-0 WER 0.2596 -> 0.1436).
    - [x] 2. Decode folds 0-4 with it (`work/asr/phowhisper_ft/train.jsonl`).
    - [~] 3. REC rollouts on folds 1-4, new ASR text (`kaggle/rec_rollouts_asr`, pushed 2026-11-15).
    - [x] 2026-11-17 REC eval fold 0 on ASR text (`kaggle/rec_eval_e5`, quyenuit24, phowhisper_ft, e5 top-7, normalised): **45.98** (hash ok 178/309) vs gold 63.23 (BGE top-5); 1c wrong value 15.9% -> 27.2%. Of 66 samples right on gold and wrong on ASR, 52 still have every spoken argument value in the ASR text (model sensitivity, fixable by training), 10 lost a value (ASR), 4 identical text.
    - [x] 4. gpt-oss rationale for gold, ASR and recovery rows (2026-11-16, `scripts/rationale_data.py`, AutoTool-style `<select>` + `<tool>` + `<args>`, spoken text with no normaliser so the model converts values itself; code-checked: gold 1085/1147 samples clean, ASR 998/1147; `scripts/rat_data.py` -> work/sft_rat 2445 train rows incl. 300 recovery, 59 val; rationale median 74 tokens per step).
    - [x] 5. One SFT from REC -> RAT: `kaggle/rat_train` finished on tquyen (306/306 steps, 9.52 GPU h, val loss 1.3007 -> 0.3862). Pulled to `work/kaggle/rat_train/out/traj_rat/adapter`. Eval fold 0 (`kaggle/rat_eval`, gold + asr_ft, normalise=False, id_guard=True, no app filter) pushed to quyenuit24 (RUNNING).
    - [x] 2026-11-22 Prepared gold + paraphrase only RAT tool-only data in `work/sft_rat_toolonly_gold_para/`: 1,115 gold, 1,575 paraphrase train rows, 59 gold val; zero ASR/recovery rows; every trained call and argument equals gold. The existing RAT tool-only checkpoint has not been retrained on this replacement data.
    - [x] 2026-11-22 `tquynlnguyn/lifemate-rat-gold-para-top7-train` COMPLETE on Kaggle T4: 2,690 rows, 337/337 steps, 6.58 GPU hours, validation loss 1.0277 -> 0.2912, no time-limit stop. Initialized from B2, one epoch, LR 5e-5, separate adapter. Source data is gold + accepted paraphrases only; E5 top-7 + contacts + sys retains every gold tool. Private input: `tquynlnguyn/lifemate-rat-gold-para-top7-data`.
    - [ ] Evaluate the new RAT checkpoint against the existing checkpoint on the same fold-0 examples with ASR and gold text, top-7 filter, and identical decoding before adding ASR-robustness data.
    - [ ] 6. RAT rollouts on folds 1-4 (greedy + 4 at T=0.7) -> DPO pairs (same prefix, chosen = gold step, rejected = own call).
    - [ ] 7. DPO from RAT (beta 0.1, lr ~5e-6, 1 epoch, + SFT term), eval fold 0 with the EDA table (watch group 1c).
  - [~] 2026-11-22 ASR few-shot correction experiment: P2 decode folds 1-4 job `tquynlnguyn/lifemate-p2-decode-train-pairs` COMPLETE with 4,027 unique clips and zero fold-0 overlap. Fold-0 P2 baseline already cached (1,009 clips, 14.2867% WER). Private input `tquynlnguyn/lifemate-asr-fewshot-input` created. The earlier TF-IDF correction job `tquynlnguyn/lifemate-asr-fewshot-correction` is CANCEL_ACKNOWLEDGED; no corrected WER is available.
   - [x] 2026-11-22 ASR to gold dense retriever: `tquynlnguyn/lifemate-asr-gold-retriever-train` COMPLETE on Kaggle T4. Trained 278M multilingual E5-base for three epochs on all 4,027 P2 ASR/gold pairs from folds 1-4; evaluated 1,009 fold-0 queries against a folds 1-4 gold bank. Top-2 oracle candidate WER: TF-IDF 63.48%, E5 base 67.40%, fine-tuned E5 65.45%. Top-3: 61.45%, 64.83%, 63.35%. Fine-tuning improves E5 but TF-IDF remains stronger on oracle WER. Metrics and examples: `reports/asr_retriever_fold0.md`; private checkpoint and samples remain under `work/kaggle/asr_gold_retriever_train/out/`.
   - [x] 2026-11-22 Few-shot correction with fine-tuned E5 retriever: `tquynlnguyn/lifemate-asr-fewshot-correction` COMPLETE on Kaggle T4 x2. With Qwen3-4B + B2, top-2 scored 18.7461% WER and top-3 19.0696% on 1,009 fold-0 clips, both worse than raw P2 at 14.2867%. Keep raw P2 and do not use this correction pass. Version 1 failed before inference due to `BatchEncoding.cuda()`; version 2 fixed it with `.to("cuda")`. Results: `reports/asr_fewshot_correction_fold0.md`; individual predictions remain ignored under `work/`.

## Stage 6 - Docker (hard deadline 2026-11-16)

- [x] Ask organisers: compute limits, external API policy. **Answered 2026-10-30** (full regs in `.claude/CLAUDE.md`): system <6B params (LLM <5B, context <=16K), scoring on 1x A30/24GB + 16-core Xeon + 64GB RAM, 30s/sample timeout, fully offline/no external API at inference, deterministic decoding, models/data from before 2026-10-26.
  - [ ] Still open: whether "offline, no external API" also binds round-1 (public_test) submissions, or only the final Docker. Ask directly; until answered, don't submit an OpenAI-API-based run to the portal.
- [ ] Offline image with weights, deterministic decoding, full public_test rehearsal, timing/VRAM, checksum, Drive link.

## Stage 7 - Private test (2026-11-20 to 11-23)

- [ ] Run the frozen Docker, validate, submit.
