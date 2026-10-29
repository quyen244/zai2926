# Artifacts - LifeMate

Tracks large outputs that live outside git: `work/` is git-ignored (per-sample data), and model/dataset dumps go to private Hugging Face repos via `scripts/hf_upload.py`.
This file itself is tracked in git, so it is the single place to look up where an artifact actually lives, from any machine.

| Artifact | Produced on | Script | Location | Status |
|---|---|---|---|---|
| ASR output: `work/asr/phowhisper/{train,public_test}.jsonl` | server rtx | `scripts/asr_run.py` | local disk on server rtx | not yet uploaded to HF |
| PhoWhisper CT2 model | server rtx | `ct2-transformers-converter` (see `.claude/CLAUDE.md`) | local disk on server rtx | not yet uploaded to HF |

Update this table whenever `hf_upload.py` is run (add the HF repo URL) or a new large output is produced on any machine.
