"""Fine-tune multilingual E5 base to retrieve gold text from PhoWhisper P2 ASR.

Train on folds 1-4. Compare retrieval before and after training on fold 0.
The retrieval bank contains only the 4,027 training gold texts.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"
MODEL = "intfloat/multilingual-e5-base"
SEED = 42


def one_file(name: str) -> Path:
    paths = list(Path("/kaggle/input").rglob(name))
    print(name, [str(p) for p in paths], flush=True)
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name}, found {len(paths)}")
    return paths[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", text.lower())).split())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = read_jsonl(one_file("real.jsonl"))
    decoded = read_jsonl(one_file("train_p2.jsonl"))
    fold0_decoded = read_jsonl(one_file("baseline.jsonl"))
    manifest = {r["audio"]: r for r in real}
    train_audio = {a for a, r in manifest.items() if r["fold"] != 0}
    fold0 = {a for a, r in manifest.items() if r["fold"] == 0}
    if len(real) != 5036 or len(train_audio) != 4027 or len(fold0) != 1009:
        raise RuntimeError("real split changed")
    if len(decoded) != 4027 or {r["audio"] for r in decoded} != train_audio:
        raise RuntimeError("P2 train transcript coverage changed")
    if len(fold0_decoded) != 1009 or {r["audio"] for r in fold0_decoded} != fold0:
        raise RuntimeError("P2 fold-0 transcript coverage changed")
    rows = [{"audio": r["audio"], "asr": r["hyp"], "gold": manifest[r["audio"]]["text"],
             "fold": manifest[r["audio"]]["fold"]} for r in decoded]
    if any(not r["asr"].strip() or not r["gold"].strip() for r in rows):
        raise RuntimeError("empty ASR or gold text")
    test = [{"audio": r["audio"], "asr": r["hyp"], "gold": manifest[r["audio"]]["text"]}
            for r in fold0_decoded]
    if any(not r["asr"].strip() or not r["gold"].strip() for r in test):
        raise RuntimeError("empty fold-0 ASR or gold text")
    print(f"rows: train={len(rows)} test_fold0={len(test)} bank={len(rows)}", flush=True)
    print("pair example:", json.dumps(rows[0], ensure_ascii=False), flush=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "jiwer==4.0.0"], check=True)
    import jiwer
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoModel, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda()
    params = sum(p.numel() for p in model.parameters())
    if params >= 400_000_000:
        raise RuntimeError(f"retriever exceeds 400M parameter budget: {params}")
    print(f"model={MODEL} parameters={params}", flush=True)

    def embed(texts: list[str], prefix: str, batch: int = 64) -> np.ndarray:
        model.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                encoded = tokenizer([prefix + t for t in texts[start:start + batch]],
                                    padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
        return np.concatenate(vectors)

    bank_gold = [r["gold"] for r in rows]
    test_asr = [r["asr"] for r in test]
    test_gold = [norm(r["gold"]) for r in test]
    normalized_bank = [norm(g) for g in bank_gold]

    def score(matrix: np.ndarray) -> tuple[dict, np.ndarray]:
        ranking = np.argsort(-matrix, axis=1, kind="stable")[:, :3]
        exact = [0, 0, 0]
        oracle_edits = [0, 0, 0]
        words = sum(len(ref.split()) for ref in test_gold)
        for i, indices in enumerate(ranking):
            edits = []
            for j, index in enumerate(indices):
                candidate = normalized_bank[int(index)]
                result = jiwer.process_words(test_gold[i], candidate)
                edits.append(result.substitutions + result.deletions + result.insertions)
                if candidate == test_gold[i]:
                    exact[j] += 1
            for k in (1, 2, 3):
                oracle_edits[k - 1] += min(edits[:k])
        metrics = {f"oracle_wer_at_{k}": round(oracle_edits[k - 1] / words, 6)
                   for k in (1, 2, 3)}
        metrics.update({f"exact_gold_at_{k}": sum(exact[:k]) for k in (1, 2, 3)})
        return metrics, ranking

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    bank_sparse = tfidf.fit_transform(bank_gold)
    tfidf_metrics, _ = score((tfidf.transform(test_asr) @ bank_sparse.T).toarray())
    base_metrics, _ = score(embed(test_asr, "query: ") @ embed(bank_gold, "passage: ").T)
    print("fold0 before training", json.dumps({"tfidf": tfidf_metrics, "e5_base": base_metrics}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for epoch in range(1, 4):
        model.train()
        order = list(range(len(rows)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), 16):
            batch = [rows[i] for i in order[start:start + 16]]
            if len(batch) < 2:
                continue
            query = tokenizer(["query: " + r["asr"] for r in batch], padding=True,
                              truncation=True, max_length=128, return_tensors="pt").to("cuda")
            passage = tokenizer(["passage: " + r["gold"] for r in batch], padding=True,
                                truncation=True, max_length=128, return_tensors="pt").to("cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                qh = model(**query).last_hidden_state
                ph = model(**passage).last_hidden_state
                q = F.normalize((qh * query["attention_mask"].unsqueeze(-1)).sum(1).float()
                                / query["attention_mask"].sum(1).unsqueeze(-1), dim=-1)
                p = F.normalize((ph * passage["attention_mask"].unsqueeze(-1)).sum(1).float()
                                / passage["attention_mask"].sum(1).unsqueeze(-1), dim=-1)
                logits = (q @ p.T) / 0.05
                same = torch.tensor([[norm(a["gold"]) == norm(b["gold"]) and i != j
                                      for j, b in enumerate(batch)] for i, a in enumerate(batch)], device="cuda")
                logits = logits.masked_fill(same, -1e4)
                loss = F.cross_entropy(logits, torch.arange(len(batch), device="cuda"))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        record = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6)}
        history.append(record)
        print("epoch", json.dumps(record), flush=True)
    model.save_pretrained(OUT / "retriever", safe_serialization=True)
    tokenizer.save_pretrained(OUT / "retriever")
    trained_metrics, ranking = score(embed(test_asr, "query: ") @ embed(bank_gold, "passage: ").T)

    examples = []
    for i, row in enumerate(test[:20]):
        examples.append({"audio": row["audio"], "asr": row["asr"], "gold": row["gold"],
                         "top3": [{"audio": rows[int(j)]["audio"], "gold": rows[int(j)]["gold"]}
                                  for j in ranking[i]]})
    (OUT / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"model": MODEL, "parameter_count": params, "train_rows": len(rows),
              "test_fold0_rows": len(test), "bank_rows": len(rows), "fold0_train_rows": 0,
              "baseline_tfidf": tfidf_metrics, "baseline_e5": base_metrics,
              "trained_e5": trained_metrics, "history": history,
              "metric_note": "Fold-0 gold is absent from the bank. Oracle WER is best retrieved gold versus fold-0 gold, not corrected-ASR WER."}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("result", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
