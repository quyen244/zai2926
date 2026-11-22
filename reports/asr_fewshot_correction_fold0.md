# ASR few-shot correction on PhoWhisper P2 fold 0

Kaggle job `tquynlnguyn/lifemate-asr-fewshot-correction` completed on 2026-11-22 using Kaggle T4 x2.

The correction model was `Qwen/Qwen3-4B` with the selected B2 LoRA adapter.

For each of 1,009 fold-0 P2 transcripts, the job retrieved either the top 2 or top 3 examples from a fine-tuned `intfloat/multilingual-e5-base` retriever.

The retrieval bank contained 4,027 ASR/gold pairs from folds 1-4, with zero fold-0 overlap.

Both variants were scored against the same raw PhoWhisper P2 output using 14,531 reference words and 2,076 baseline edits, equivalent to 14.2867% WER.

| Variant | Edits | WER | Change from raw P2 |
|---|---:|---:|---:|
| Raw PhoWhisper P2 | 2,076 | 14.2867% | - |
| Few-shot top 2 | 2,724 | 18.7461% | +4.4594 percentage points |
| Few-shot top 3 | 2,771 | 19.0696% | +4.7829 percentage points |

Few-shot correction did not improve WER, and top 2 was slightly better than top 3 while still substantially worse than raw P2.

Keep raw PhoWhisper P2 as the selected transcript; do not apply this Qwen + B2 correction pass.

The first kernel version failed before inference because the tokenizer `BatchEncoding` was moved with `.cuda()`; version 2 fixed this with `.to("cuda")` and completed all 2,018 correction variants.

Aggregate metrics: `work/kaggle/asr_fewshot_correction_output/out/metrics.json`.

Per-sample predictions remain under `work/` and are not committed.
