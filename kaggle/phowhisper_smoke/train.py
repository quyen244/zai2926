from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

MODEL_ID = "vinai/PhoWhisper-large"
DATA = Path(os.environ.get("ASR_DATA", "/kaggle/input/phowhisper-smoke-audio"))
OUT = Path(os.environ.get("ASR_OUT", "/kaggle/working/out"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], processor):
        self.items = []
        for row in rows:
            audio, sr = sf.read(DATA / row["audio"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
            labels = processor.tokenizer(row["text"], add_special_tokens=True).input_ids
            self.items.append({"input_features": features, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        x = torch.tensor(np.stack([r["input_features"] for r in rows]), dtype=torch.float32)
        labels = [r["labels"] for r in rows]
        padded = self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        padded["input_ids"][padded["attention_mask"] == 0] = -100
        if (padded["input_ids"][:, 0] == self.processor.tokenizer.bos_token_id).all():
            padded["input_ids"] = padded["input_ids"][:, 1:]
        return {"input_features": x, "labels": padded["input_ids"]}


def main():
    t0 = time.perf_counter()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    rows = [json.loads(x) for x in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"], bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    train_data = SpeechDataset(rows[:112], processor)
    val_data = SpeechDataset(rows[112:], processor)
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "checkpoint"), max_steps=MAX_STEPS,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-4, warmup_steps=2,
        fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=True,
        eval_strategy="steps", eval_steps=MAX_STEPS, save_strategy="steps", save_steps=MAX_STEPS,
        logging_steps=1, report_to="none", predict_with_generate=False,
        dataloader_num_workers=2, remove_unused_columns=False,
        save_total_limit=1, seed=42, data_seed=42,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_data,
                             eval_dataset=val_data, data_collator=Collator(processor), processing_class=processor)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    if trainer.is_world_process_zero():
        final = OUT / "adapter"
        trainer.save_model(final)
        processor.save_pretrained(final)
        stats = {"model": MODEL_ID, "samples": len(rows), "train_samples": len(train_data),
                 "smoke_steps": MAX_STEPS, "train_seconds": elapsed,
                 "seconds_per_step": elapsed / max(1, MAX_STEPS),
                 "samples_per_second_global": result.metrics.get("train_samples_per_second"),
                 "peak_vram_gib": torch.cuda.max_memory_reserved() / 1024**3,
                 "train_metrics": result.metrics, "elapsed_total_seconds": time.perf_counter()-t0}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("SMOKE_STATS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
