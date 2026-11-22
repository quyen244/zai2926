"""Continue correction-base from its epoch-2 checkpoint through epoch 5 on Kaggle."""
import json, re, tarfile, unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

MODEL_ID = "nrl-ai/vn-spell-correction-base"
ROOT = Path("/kaggle/working")
DATA = Path("/kaggle/input")
OUT = ROOT / "out"
NUM = {"không":"0", "một":"1", "hai":"2", "ba":"3", "bốn":"4", "năm":"5", "sáu":"6", "bảy":"7", "tám":"8", "chín":"9", "mười":"10"}

def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(NUM.get(w, w) for w in s.split())

def edits(ref, hyp):
    a, b = ref.split(), hyp.split()
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]

def wer(rows, field):
    total_words = total_edits = 0
    for r in rows:
        ref, hyp = norm(r["ref"]), norm(r[field])
        total_words += len(ref.split())
        total_edits += edits(ref, hyp)
    return total_edits / max(total_words, 1)

def predict(model, tok, rows, batch=16):
    model.eval()
    device = next(model.parameters()).device
    result = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i+batch]
            x = tok([r["hyp"] for r in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            y = model.generate(**x, num_beams=4, max_new_tokens=192)
            result.extend(tok.batch_decode(y, skip_special_tokens=True))
            print(f"infer {min(i+len(chunk),len(rows))}/{len(rows)}", flush=True)
    return result

def fine_tune(model, tok, rows, val, initial_wer):
    from torch.utils.data import Dataset, DataLoader
    class Pairs(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            r = self.items[i]
            x = tok(r["hyp"], truncation=True, max_length=256)
            y = tok(text_target=r["ref"], truncation=True, max_length=192)
            return x["input_ids"], x["attention_mask"], y["input_ids"]
    def collate(batch):
        xs = tok.pad({"input_ids":[x[0] for x in batch], "attention_mask":[x[1] for x in batch]}, padding=True, return_tensors="pt")
        n = max(len(x[2]) for x in batch)
        labels = torch.tensor([x[2] + [-100] * (n-len(x[2])) for x in batch])
        return {**xs, "labels": labels.to(xs["input_ids"].device)}
    device = next(model.parameters()).device
    loader = DataLoader(Pairs(rows), batch_size=8, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_epoch, best_wer = 2, initial_wer
    epoch_results = []
    for epoch in range(3, 6):
        model.train()
        loss_sum = 0
        for step, batch in enumerate(loader, 1):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            if step % 25 == 0: print(f"sft epoch={epoch} step={step}/{len(loader)} loss={loss_sum/step:.4f}", flush=True)
        mean_loss = loss_sum / max(len(loader), 1)
        predictions = predict(model, tok, val)
        for r, y in zip(val, predictions):
            r["candidate"] = y
        epoch_wer = wer(val, "candidate")
        epoch_results.append({"epoch": epoch, "train_loss": mean_loss, "fold0_wer": epoch_wer})
        (OUT / f"epoch{epoch}_predictions.jsonl").write_text(
            "".join(json.dumps({"id": r["id"], "ref": r["ref"], "hyp": r["hyp"], "corrected": r["candidate"]}, ensure_ascii=False) + "\n" for r in val),
            encoding="utf-8")
        print(f"epoch={epoch} train_loss={mean_loss:.4f} fold0_wer={epoch_wer:.6f}", flush=True)
        if epoch_wer < best_wer:
            best_epoch, best_wer = epoch, epoch_wer
            best_dir = OUT / "best_model"
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            print(f"saved best checkpoint at epoch {epoch}", flush=True)
    return epoch_results, best_epoch, best_wer

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = next(DATA.rglob("examples.tar.gz"), None)
    if archive:
        with tarfile.open(archive) as t: t.extractall(ROOT / "data")
        data_path = ROOT / "data/examples.jsonl"
    else: data_path = next(DATA.rglob("examples.jsonl"))
    rows = [json.loads(x) for x in data_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "eval"]
    if not torch.cuda.is_available():
        raise SystemExit("Kaggle GPU is required")
    if len(train) != 1236 or len(val) != 309:
        raise SystemExit(f"unexpected split sizes: train={len(train)}, heldout={len(val)}")
    checkpoint = next((p.parent for p in DATA.rglob("model.safetensors")
                       if "lifemate-spell-correction-base-sft" in str(p) and p.parent.name == "vn-spell-correction-base-sft"), None)
    if checkpoint is None:
        raise SystemExit("epoch-2 correction-base model is missing from Kaggle kernel input")
    print(f"resuming epoch-2 checkpoint {checkpoint}; train={len(train)} heldout={len(val)}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
    if len(tok) != model.config.vocab_size or tok.pad_token_id != model.config.pad_token_id:
        raise SystemExit(f"tokenizer/model mismatch: tokens={len(tok)}, model vocab={model.config.vocab_size}, pad={tok.pad_token_id}/{model.config.pad_token_id}")
    start_predictions = predict(model, tok, val)
    for r, y in zip(val, start_predictions):
        r["epoch2"] = y
    raw_wer, epoch2_wer = wer(val, "hyp"), wer(val, "epoch2")
    print(f"raw P2 WER={raw_wer:.6f}; resumed epoch-2 WER={epoch2_wer:.6f}", flush=True)
    epoch_results, best_epoch, best_wer = fine_tune(model, tok, train, val, epoch2_wer)
    metrics = {"model": MODEL_ID, "eval_split": "CV fold 0", "eval_n": len(val), "train_n": len(train),
               "raw_p2_wer": raw_wer, "epoch2_wer": epoch2_wer, "epochs3_to_5": epoch_results,
               "best_epoch": best_epoch, "best_wer": best_wer, "gain_vs_raw_p2": raw_wer - best_wer,
               "best_model_source": "out/best_model" if best_epoch > 2 else str(checkpoint),
               "optimizer_resumed": False}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__": main()
