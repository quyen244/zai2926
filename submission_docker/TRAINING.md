# Training code

All jobs ran as Kaggle kernels (T4 x2), each with its own `kaggle/<job>/` folder: `run.py` is the job, `pack.py` builds its input dataset, and `kernel-metadata.json` is its Kaggle config.
Library code lives in `lifemate/`, shipped in `/code/lifemate`.
Data scripts are in `scripts/`, and LLM prompts used only for data generation are in `prompts/`.
Base models (downloaded from the Hugging Face Hub) are Qwen/Qwen3-4B, vinai/PhoWhisper-large and intfloat/multilingual-e5-small.
Splits: `splits/cv5.json`. Fold 0 is held out for evaluation, and folds 1-4 are used for training.

1. ASR: `kaggle/phowhisper_train*` trains a LoRA on PhoWhisper-large using the train audio (real clips plus synthetic clips from `kaggle/asr_synth`). `kaggle/asr_decode` merges it, converts it to CTranslate2 and decodes every train clip.
2. Agent SFT data: `scripts/plan_data.py` and `scripts/recovery_data.py` build trajectories from the gold tool calls (train folds 1-4), and recovery rows teach the model to look up an ID after an ID-guard error.
3. LLM SFT: `lifemate/sft_train.py` (unsloth, 4-bit base, LoRA r16), run by `kaggle/plan_sft_train`, then `kaggle/b2_train`, then `kaggle/recovery_train` (REC = the adapter used at inference; lr 1e-5, init from the previous adapter).
4. App filter: `scripts/train_app_embedder.py` (run by `kaggle/embed_train`): multilingual-e5-small, listwise InfoNCE over installed apps, 5 epochs, lr 2e-5, seed 42.

Seeds: 0 in SFT (`--seed`), 42 in the embedder. Evaluation uses fold 0 with `lifemate/scoring.py`.
