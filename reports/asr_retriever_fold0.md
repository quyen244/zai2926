# PhoWhisper P2 ASR to gold retriever

Kaggle job `tquynlnguyn/lifemate-asr-gold-retriever-train` completed on 2026-11-22.

`intfloat/multilingual-e5-base` has 278,043,648 parameters.

The model trained for three epochs on 4,027 P2 ASR/gold pairs from folds 1-4.

Evaluation used 1,009 fold-0 ASR queries and a retrieval bank of 4,027 gold transcripts from folds 1-4.

No fold-0 gold transcript was included in training or the retrieval bank.

| Retriever | Oracle WER @1 | Oracle WER @2 | Oracle WER @3 | Exact gold in top 1 | Top 2 | Top 3 |
|---|---:|---:|---:|---:|---:|---:|
| Character TF-IDF | 69.44% | 63.48% | 61.45% | 21 | 28 | 32 |
| E5-base before fine-tuning | 73.81% | 67.40% | 64.83% | 19 | 25 | 28 |
| E5-base after fine-tuning | 72.49% | 65.45% | 63.35% | 23 | 31 | 34 |

Fine-tuning improved E5's oracle WER by 1.32, 1.95, and 1.47 percentage points at top 1, top 2, and top 3.

Character TF-IDF still had lower oracle WER than fine-tuned E5 by 3.04, 1.97, and 1.90 percentage points.

Fine-tuned E5 retrieved the exact gold transcript for 23, 31, and 34 fold-0 queries in the top 1, top 2, and top 3.

Oracle WER means the lowest WER among the retrieved gold candidates, chosen using the fold-0 gold transcript.

It is an upper bound for selecting one candidate from the retrieved set, not the WER of an LLM correction pass.

The correction model has not yet been run with this fine-tuned retriever.

Files: `work/kaggle/asr_gold_retriever_train/out/metrics.json`, `examples.json`, and `retriever/`.
