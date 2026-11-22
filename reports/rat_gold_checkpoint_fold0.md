# RAT gold + paraphrase checkpoint evaluation on fold 0

Kaggle kernel `tquynlnguyn/lifemate-rat-gold-checkpoint-eval` version 2 completed on 2026-11-22 using two GPUs.

This evaluates only the new `Qwen3-4B-trajRAT-gold` adapter, initialized from B2 and trained on gold plus accepted paraphrases with top-7 filtering.

Evaluation used the 309 fold-0 samples with either gold user text or the matching PhoWhisper P2 transcript.

Both variants used the same decoding, six-step cap, ID guard, top-7 app filter, and `normalise=False` setting.

The score is the official mean per-sample tool-call F1 with the state-hash match gate, reported on a 0-100 scale.

| Input text | Score | Hash match | Samples |
|---|---:|---:|---:|
| Gold | 58.5437 | 208/309 (67.31%) | 309 |
| PhoWhisper P2 | 43.3164 | 167/309 (54.05%) | 309 |

The ASR input score is 15.2273 points below the gold-input score for this checkpoint.

The fine-tuned E5 top-7 filter retained 455/498 gold apps (91.37%) in both variants, and retained every gold app for 244/309 samples.

There were 618 successful runs, zero failed runs, and zero samples skipped by the time limit.

Kernel execution took 6,167 seconds for inference and 6,855 seconds overall.

This is a one-checkpoint evaluation, so it does not establish whether the new checkpoint improves over the previous RAT checkpoint.

The first kernel version failed before inference because its input dataset was unavailable; version 2 used a newly uploaded private dataset and completed.

Aggregate output: `work/rat_gold_checkpoint_eval_latest/scores.json`.

Per-sample predictions and runtime logs remain under ignored `work/`.
