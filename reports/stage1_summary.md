# Stage 0 + 1 summary (2026-10-28)

Details: `stage0.md`, `stage1_labels.md`, `stage1_taxonomy.md`, `stage1_asr.md`.

## Stage 0 - harness (done, tested)

- `lifemate/scoring.py` re-implements the official metric; 19 unit/integration tests pass (`python -m pytest tests`).
- Gold replay: 1545/1545 train samples reproduce `final_state` and `tool_results`; gold scores itself 100.00.
- Floor: submitting `[]` everywhere scores 7.18 on train.
- Splits: `splits/cv5.json` (stratified 5 folds) and `splits/leave_app_out.json` (9 train-only apps held out, 241 val samples).
- Submission validator checks format, ids and schema of every call.
- Scorer assumptions where the rules are silent (empty values = absent, exact equality, whitespace tokens for token-F1) are listed in `lifemate/scoring.py`.

## Stage 1 - what the data says

1. **Gold style matters for precision.** Read tools are mostly called with `{}`; `contacts.search` uses the name exactly as spoken (capitalised, 96.6% verbatim in the transcript).
2. **25% of the IDs written come from history**, not from a read in the same turn; the gold then writes without reading again.
3. **Asking happens after an ambiguous search** (74 cases follow a `contacts.search` with 2+ results) or immediately when a parameter is missing (60).
4. **Empty gold is two things**: no suitable app (~80) and the speaker cancelling (~31).
5. **Shared apps have identical schemas** in train and public; 11 public apps are unseen, so app generalisation is the main modelling risk.
6. **Prompt size**: tool schemas of the installed apps are ~10k characters per sample (median), history ~1k.
7. **Every earlier user turn of public_test is audio only** (1299 turns), so ASR runs on history too.

## Stage 1.2 - ASR is the largest risk found so far

| model | clips | WER | gold-arg recall | hallucination | RTF |
|---|---|---|---|---|---|
| Whisper large-v3 | 400 | 0.297 | 0.492 | 5.8% | 0.31 |
| Whisper large-v3-turbo | 400 | 0.297 | 0.453 | 3.5% | 0.25 |
| PhoWhisper-large | 400 | **0.246** | **0.570** | **1.0%** | 0.34 |

- PhoWhisper-large is the best on every column except speed, and the best on every dialect/phenomenon slice; it is the working choice for Stage 2.
- Even with it, only about half of the gold argument strings that are spoken (names, places, queries) survive ASR verbatim. Those are non-fuzzy arguments, so each miss costs the hash or the call match.
- About 6% of clips get a **whole-utterance hallucination** (YouTube-style filler such as a "thanks for watching" line instead of the real request).
  Signal level does not explain it (same peak and RMS as normal clips); hallucinated clips have longer leading silence (0.42 s vs 0.22 s) and are longer.
  Peak normalisation or VAD alone removes it on 11 of 25 clips.
  Root cause, per quyen (listened to the affected clips): the audio itself is hard to hear/understand in those clips, not a preprocessing artifact.
  On PhoWhisper, the model used from Stage 2 on, the rate is 1.0% (4/400); accepted as is.
- Southern dialect clips have the worst WER (0.39 for large-v3).

## Not done in this stage

- Listening pass (1.1): the sheet is ready at `work/listening/index.html` (170 clips: 120 train + 50 public); it needs a human.
- LLM tagging (1.4): pipeline, taxonomy (`taxonomy/taxonomy.json`) and rate limiting are ready and tested on 6 samples; it waits for the team's prompts in `prompts/`.
- PhoWhisper hallucinations (4 clips): closed, see Stage 1.2 above.

## Gate for Stage 2

Choose the ASR (or fine-tune it) using gold-arg recall, not WER alone. The hallucination failure mode zeroes the whole sample, but on PhoWhisper it is down to 1.0% and traced to hard-to-hear audio, so it no longer blocks Stage 2.
