"""PhoWhisper-large LoRA, one epoch split over two Kaggle jobs (torchrun, 2 x T4, one process per GPU).

Train set: real recordings + TTS batch 1 + TTS batch 2, CV folds 1-4 only, one fixed order (Trainer sampler, seed 42).
Val set: 300 real fold-0 recordings (sorted by id), loss only.
PART=1 trains steps [0, half) and saves a full checkpoint (adapter, optimizer, scheduler, RNG) at `half`.
PART=2 resumes that checkpoint: same lr schedule, the first `half` steps are skipped without loading audio.
Either part saves and stops early at MAX_HOURS. Labels carry the <|vi|><|transcribe|> prefix.

Env: REAL_DIR SYN1_DIR SYN2_DIR (extracted audio + manifests), DATA (dataset dir), OUT, PART, CKPT (part 2), MAX_HOURS, LIMIT.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          TrainerCallback)

MODEL_ID = "vinai/PhoWhisper-large"
DATA = Path(os.environ["DATA"])
OUT = Path(os.environ["OUT"])
PART = int(os.environ["PART"])
MAX_HOURS = float(os.environ.get("MAX_HOURS", "10.5"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # smoke: use only this many train rows
BS, ACCUM, LR = 4, 2, 1e-4
SR = 16000


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def items() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    cv = json.loads((DATA / "cv5.json").read_text(encoding="utf-8"))
    real_dir, s1, s2 = Path(os.environ["REAL_DIR"]), Path(os.environ["SYN1_DIR"]), Path(os.environ["SYN2_DIR"])
    real = jsonl(DATA / "real.jsonl")
    syn1 = jsonl(DATA / "synth1.jsonl")
    syn2 = jsonl(Path(os.environ.get("SYN2_MANIFEST") or s2 / "manifest.jsonl"))
    train = [(str(real_dir / r["audio"]), r["text"]) for r in real if r["fold"] != 0]
    train += [(str(s1 / r["audio"]), r["text"]) for r in syn1 if r["fold"] != 0]
    train += [(str(s2 / r["audio"]), r["text"]) for r in syn2 if cv[r["source_id"].rsplit("_u", 1)[0]] != 0]
    val = [(str(real_dir / r["audio"]), r["text"]) for r in sorted(real, key=lambda r: r["id"]) if r["fold"] == 0][:300]
    train.sort()
    random.Random(42).shuffle(train)  # fixed base order; the Trainer sampler (seed 42) shuffles on top, same in both parts
    return (train[:LIMIT] if LIMIT else train), val


class Speech(torch.utils.data.Dataset):
    def __init__(self, rows, processor):
        self.rows, self.p = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        audio = audio[: 30 * SR]
        feats = self.p.feature_extractor(audio, sampling_rate=SR, return_tensors="np").input_features[0]
        labels = self.p.tokenizer(text).input_ids[:440]
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, processor):
        self.p = processor

    def __call__(self, rows):
        x = torch.tensor(np.stack([r["input_features"] for r in rows]), dtype=torch.float16)
        pad = self.p.tokenizer.pad({"input_ids": [r["labels"] for r in rows]}, padding=True, return_tensors="pt")
        y = pad["input_ids"].masked_fill(pad["attention_mask"] == 0, -100)
        if (y[:, 0] == self.p.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            y = y[:, 1:]  # the model prepends decoder_start_token itself
        return {"input_features": x, "labels": y}


class Stop(TrainerCallback):
    def __init__(self, half: int):
        self.half, self.t0 = half, time.time()

    def on_step_end(self, args, state, control, **kw):
        if PART == 1 and state.global_step >= self.half:
            control.should_save = control.should_training_stop = True
        if time.time() - self.t0 > MAX_HOURS * 3600:
            print(f"MAX_HOURS reached at step {state.global_step}", flush=True)
            control.should_save = control.should_training_stop = True


def main():
    t0 = time.time()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    train, val = items()
    steps = math.ceil(len(train) / (BS * world * ACCUM))
    half = steps // 2
    if rank == 0:
        print(f"PART {PART}: train rows {len(train)}, val rows {len(val)}, steps {steps}, part 1 ends at {half}", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none"))
    model.enable_input_require_grads()
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "checkpoint"), max_steps=steps, per_device_train_batch_size=BS, per_device_eval_batch_size=BS,
        gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=max(1, steps // 50), lr_scheduler_type="linear",
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, eval_strategy="no", save_strategy="no", logging_steps=10, report_to="none",
        predict_with_generate=False, dataloader_num_workers=2, remove_unused_columns=False, seed=42, data_seed=42)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=Speech(train, processor), eval_dataset=Speech(val, processor),
                             data_collator=Collator(processor), processing_class=processor, callbacks=[Stop(half)])
    ckpt = os.environ.get("CKPT") if PART == 2 else None
    result = trainer.train(resume_from_checkpoint=ckpt)
    ev = trainer.evaluate()
    if trainer.is_world_process_zero():
        if PART == 2 or trainer.state.global_step >= steps:
            trainer.save_model(OUT / "adapter")
            processor.save_pretrained(OUT / "adapter")
        stats = {"part": PART, "train_rows": len(train), "val_rows": len(val), "steps_total": steps, "half": half,
                 "global_step": trainer.state.global_step, "val_loss": ev.get("eval_loss"), "train_metrics": result.metrics,
                 "peak_vram_gib": torch.cuda.max_memory_reserved() / 1024**3, "hparams": {"bs": BS, "accum": ACCUM, "lr": LR,
                 "world": world, "lora": "r8 a16 q,v"}, "seconds": time.time() - t0}
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("STATS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
