"""QLoRA SFT of Qwen3-4B on one plan-ablation dataset (A = "plan", B = "traj"); one process per GPU.

  CUDA_VISIBLE_DEVICES=0 python -m lifemate.sft_train --data plan --out adapters/plan
  CUDA_VISIBLE_DEVICES=1 python -m lifemate.sft_train --data traj --out adapters/traj
  ... --max-steps 10 --limit 64   # smoke test: measures tokens/s and memory

Every setting is shared by the two experiments (only --data differs): base model, tokenizer, data order (seed),
4-bit NF4 base + fp16 compute (T4 has no bf16), LoRA r16/a32 on all attention and MLP projections (dropout 0 with the
unsloth backend, the default; 0.05 with --backend hf),
lr 1e-4 cosine with 3% warmup, batch 1 x grad accum 16, max length 10240, 1 pass over the train rows (each sample
appears twice: reference text and ASR text), gradient checkpointing, paged 8-bit AdamW, seed 0.
Loss masking: lifemate.sft (trained assistant turns only); the loss is computed on those positions only (selective_loss)
and SDPA is held to the memory-efficient kernel on a T4 (t4_attention). Writes <out>/adapter (PEFT), <out>/log.jsonl (train and
val loss per logging step) and <out>/stats.json (rows, tokens, supervised tokens, wall time, peak memory, throughput).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .sft import IGNORE, encode

ROOT = Path(__file__).resolve().parent.parent


def load_rows(name: str, part: str, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "work" / "sft" / f"{name}.{part}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


class Rows(torch.utils.data.Dataset):
    def __init__(self, encoded: list[dict]):
        self.x = encoded

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.x[i]["input_ids"], "labels": self.x[i]["labels"]}


LAST = {"tokens": 0}  # length of the batch just collated, for the per-row speed log


def collate(batch: list[dict]) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    LAST["tokens"] = n * len(batch)
    ids = torch.full((len(batch), n), 0, dtype=torch.long)
    lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, b in enumerate(batch):
        ids[k, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
        lab[k, : len(b["labels"])] = torch.tensor(b["labels"])
        att[k, : len(b["input_ids"])] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def selective_loss(lm, inputs: dict, num_items_in_batch=None) -> torch.Tensor:
    """Causal-LM loss computed only where a label exists: the lm_head runs on the ~100-200 supervised positions instead
    of every position (full logits at 10240 tokens x 151936 vocab are 6.2 GB in fp32, more than a T4 has left).
    Same value as the model's own loss: sum of token CE / num_items_in_batch (the supervised tokens of the whole
    accumulated batch, which Trainer counts), or the mean over this row when that count is not given (evaluation)."""
    labels = inputs["labels"]
    h = lm.model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    keep = tgt != IGNORE
    logits = lm.lm_head(h[:, :-1][keep]).float()
    loss = F.cross_entropy(logits, tgt[keep], reduction="sum")
    n = num_items_in_batch if num_items_in_batch is not None else keep.sum().clamp(min=1)
    return loss / n


def t4_attention() -> contextlib.AbstractContextManager:
    """On GPUs below sm80 (T4 = sm75: no FlashAttention), make SDPA use the memory-efficient kernel. A batch-1 row has
    no padding, so transformers drops the all-ones mask and passes enable_gqa=True; on CUDA only the flash and math
    kernels take GQA, so SDPA fell back to math, which builds the full n x n attention matrix (OOM at step 1, Kaggle,
    2026-11-08). Repeating the KV heads instead keeps is_causal and lets the efficient kernel run; restricting SDPA to
    that kernel makes any other fallback fail loudly instead of running out of memory."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 8:
        return contextlib.nullcontext()
    import transformers.integrations.sdpa_attention as sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sdpa.use_gqa_in_sdpa = lambda *a, **k: False
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["plan", "traj"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-len", type=int, default=10240)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the N longest train rows (worst-case memory)")
    ap.add_argument("--evals", type=int, default=4, help="val-loss evaluations over the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0, help="stop training cleanly after this many hours of training "
                    "(Kaggle kills a session at 12 h and keeps nothing), then evaluate and save the adapter; 0 = no limit")
    ap.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth",
                    help="unsloth: FastLanguageModel 4-bit + its offloaded gradient checkpointing and attention kernels "
                         "(Kaggle notebook Qwen3 (4B) Tool_Calling); hf: transformers + bitsandbytes + peft")
    a = ap.parse_args()

    if a.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers so its patches apply)
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    set_seed(a.seed)
    random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    train = [encode(r, tok, a.max_len) for r in load_rows(a.data, "train", 0)]
    if a.limit:  # smoke test: the LONGEST rows, so the peak memory measured is the worst case of the full run
        train = sorted(train, key=lambda x: -x["n_tokens"])[: a.limit]
    val = [encode(r, tok, a.max_len) for r in load_rows(a.data, "val", 0 if not a.limit else 16)]
    print(f"[{a.data}] encoded {len(train)} train / {len(val)} val rows in {time.time() - t0:.0f}s; "
          f"tokens {sum(x['n_tokens'] for x in train)}, supervised {sum(x['n_supervised'] for x in train)}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if a.backend == "unsloth":
        # Labels stay ours (lifemate.sft: only the final turn's assistant messages); unsloth's train_on_responses_only
        # would also train every history assistant turn. dtype None = float16 on a T4. Dropout 0 is unsloth's fast path.
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_len, load_in_4bit=True, dtype=None)
        model = FastLanguageModel.get_peft_model(model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias="none", target_modules=targets,
                                                 use_gradient_checkpointing="unsloth", random_state=a.seed)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()

    steps_total = a.max_steps if a.max_steps > 0 else int(len(train) * a.epochs / a.accum)
    every = max(1, steps_total // a.evals)
    log = open(a.out / "log.jsonl", "w", encoding="utf-8")

    class Log(TrainerCallback):
        t_row = None
        rows = 0
        tokens = 0  # tokens of the rows actually trained (an early stop trains fewer)
        busy = 0.0  # seconds spent in train forward/backward rows, evaluations excluded
        stopped = False

        def on_log(self, args, state, control, logs=None, **kw):
            log.write(json.dumps({"step": state.global_step, "time_s": round(time.time() - t_train, 1), **(logs or {})}) + "\n")
            log.flush()

        def row_done(self) -> None:  # one micro-batch (= one row) forward + backward
            now = time.time()
            dt = now - (self.t_row or t_train)
            self.t_row, self.rows, self.tokens, self.busy = now, self.rows + 1, self.tokens + LAST["tokens"], self.busy + dt
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
            print(f"[{a.data}] row {self.rows}: {LAST['tokens']} tokens in {dt:.1f}s = {LAST['tokens'] / max(dt, 1e-6):.0f} tok/s | peak {mem:.2f} GB", flush=True)

        def on_substep_end(self, args, state, control, **kw):
            self.row_done()

        def on_step_end(self, args, state, control, **kw):  # the last micro-batch of a step ends here, not in on_substep_end
            self.row_done()
            if a.max_hours and time.time() - t_train > a.max_hours * 3600:
                self.stopped = control.should_training_stop = True
                print(f"[{a.data}] time limit {a.max_hours} h reached at step {state.global_step}: stopping", flush=True)

        def on_evaluate(self, args, state, control, **kw):  # the next row's timer starts after the evaluation
            self.t_row = time.time()

    args = TrainingArguments(
        output_dir=str(a.out / "ckpt"), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, num_train_epochs=a.epochs, max_steps=a.max_steps,
        fp16=True, optim="paged_adamw_8bit", logging_steps=1, eval_strategy="steps", eval_steps=every, save_strategy="no",
        report_to=[], prediction_loss_only=True, label_names=["labels"], seed=a.seed, data_seed=a.seed, dataloader_num_workers=0, remove_unused_columns=False, max_grad_norm=1.0,
        # unsloth wires its own (offloaded) checkpointing in get_peft_model; HF's must stay off then
        gradient_checkpointing=a.backend == "hf", gradient_checkpointing_kwargs={"use_reentrant": False} if a.backend == "hf" else None)
    class SelectiveTrainer(Trainer):
        def __init__(self, *x, **k):
            super().__init__(*x, **k)
            self.model_accepts_loss_kwargs = True  # the loss below is already divided by num_items_in_batch

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = selective_loss(self.accelerator.unwrap_model(model).get_base_model(), inputs, num_items_in_batch)
            return (loss, {}) if return_outputs else loss

    # hf: our selective loss (full logits do not fit a T4); unsloth computes its own memory-light loss from the labels
    cls = SelectiveTrainer if a.backend == "hf" else Trainer
    cb = Log()
    trainer = cls(model=model, args=args, train_dataset=Rows(train), eval_dataset=Rows(val), data_collator=collate, callbacks=[cb])
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    with t4_attention() if a.backend == "hf" else contextlib.nullcontext():
        trainer.train()
        final_eval = trainer.evaluate()
    wall = time.time() - t_train
    model.save_pretrained(a.out / "adapter")
    tok.save_pretrained(a.out / "adapter")
    stats = {"data": a.data, "model": a.model, "rows": len(train), "steps": trainer.state.global_step, "steps_planned": steps_total,
             "stopped_by_time_limit": cb.stopped, "rows_seen": cb.rows, "tokens_seen": cb.tokens,
             "tokens_per_row": sum(x["n_tokens"] for x in train) / len(train), "supervised_per_row": sum(x["n_supervised"] for x in train) / len(train),
             "train_wall_s": round(wall), "gpu_hours": round(wall / 3600, 2), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
             "tokens_per_s": round(cb.tokens / max(cb.busy, 1e-6), 1),
             "final_val_loss": final_eval.get("eval_loss"), "gpu": torch.cuda.get_device_name(0), "backend": a.backend, "hparams": vars(a) | {"out": str(a.out)}}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    print(f"[{a.data}] done: {json.dumps(stats, default=str)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
