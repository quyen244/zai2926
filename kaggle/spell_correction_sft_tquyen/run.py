"""Fine-tune Vietnamese correction on PhoWhisper P2 hypotheses, with a fold-0 WER gate."""
import json, re, tarfile, unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "nrl-ai/vn-spell-correction-small"
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

def fine_tune(model, tok, rows):
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
    model.train()
    for epoch in range(2):
        loss_sum = 0
        for step, batch in enumerate(loader, 1):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            if step % 25 == 0: print(f"sft epoch={epoch+1} step={step}/{len(loader)} loss={loss_sum/step:.4f}", flush=True)
        print(f"sft epoch={epoch+1} mean_loss={loss_sum/max(len(loader),1):.4f}", flush=True)

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
    print(f"GPU={torch.cuda.is_available()} train={len(train)} heldout={len(val)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
    base = predict(model, tok, val)
    for r, y in zip(val, base): r["corrected"] = y
    before, after = wer(val, "hyp"), wer(val, "corrected")
    metrics = {"model": MODEL_ID, "eval_split": "CV fold 0", "eval_n": len(val), "train_n": len(train), "wer_asr": before, "wer_model": after, "absolute_gain": before-after, "sft_ran": False}
    print(json.dumps(metrics), flush=True)
    if before - after < 0.005:
        print("gain below 0.005 absolute; running domain SFT on folds 1-4", flush=True)
        fine_tune(model, tok, train)
        tuned = predict(model, tok, val)
        for r, y in zip(val, tuned): r["sft"] = y
        tuned_wer = wer(val, "sft")
        metrics.update({"sft_ran": True, "wer_sft": tuned_wer, "sft_gain_vs_asr": before-tuned_wer, "sft_gain_vs_model": after-tuned_wer})
        model.save_pretrained(OUT / "vn-spell-correction-small-sft")
        tok.save_pretrained(OUT / "vn-spell-correction-small-sft")
        metrics["saved_model"] = "out/vn-spell-correction-small-sft"
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "fold0_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in val), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__": main()
