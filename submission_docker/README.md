# LifeMate - Zalo AI Challenge 2026 solution

Voice request -> ASR -> tool-calling agent on the split's zaic_tools.

## Inference (`predict.sh` -> `predict.py` -> `runtime.py`)

- **ASR**: PhoWhisper-large (vinai/PhoWhisper-large, ~1.55B params) fine-tuned with LoRA on the train audio (folds 1-4), merged and converted to CTranslate2 float16 (`saved_models/phowhisper-ft-ct2`). Vietnamese, beam 5, no previous-text conditioning. Every final clip and every history user clip is transcribed.
- **App filter**: intfloat/multilingual-e5-small (~118M params) fine-tuned with a listwise InfoNCE loss over each sample's installed apps (`saved_models/app_embedder_e5`). It keeps the 7 installed apps closest to the request, plus `contacts`; `sys.*` is always offered.
- **LLM**: Qwen3-4B (~4.0B params, context limited to 16,384 tokens) + LoRA rank 16 "REC" (`saved_models/rec_adapter`), served by a local vLLM (loopback only), temperature 0, seed 0, thinking off.
- **Agent loop** (`lifemate/agent.py`): the system prompt holds the user context; the tools come from the catalog of the installed apps; a number/time normaliser runs on the text. Each step, the model returns native tool calls, which run on one `zaic_tools.ToolEnv` per sample (validate/call only), and the result goes back to the model. The loop stops at 6 steps or when the model replies without a call. An ID guard blocks calls whose IDs never appeared in the conversation and asks the model to look them up.
- Total parameters: about 5.7B (< 6B). The LLM has 4.0B (< 5B).
- Seeds are fixed (42 for Python/NumPy/PyTorch, 0 for vLLM), and decoding is deterministic.
- `<split>_environment/` is never read: the environment is reached only through zaic_tools.

## Training

See `training_code/README.md`.
