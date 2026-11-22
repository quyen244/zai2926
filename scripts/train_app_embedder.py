"""Train and tune a lightweight embedding model (intfloat/multilingual-e5-small, 118M params) for App Filtering.

Strategies:
  A: Core apps (contacts, sys) always kept (zero cost).
  B: Fine-tune multilingual-e5-small on train folds 1-4 with InfoNCE loss + hard negatives (uncalled installed apps).
  C: Dynamic Thresholding + Top-K tuning on held-out Fold 0 to guarantee >= 99% recall.

Usage:
  python scripts/train_app_embedder.py --smoke                 # quick CPU dry run on 20 samples
  python scripts/train_app_embedder.py --epochs 5 --eval-only  # evaluate pre-trained / fine-tuned model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lifemate.data import load_catalog, load_samples
from lifemate.normalize import normalize_text

MODEL_NAME = "intfloat/multilingual-e5-small"
OUT_DIR = PROJECT / "work" / "models" / "app_embedder_e5"


def enriched_app_doc(a: dict) -> str:
    app_id = a["id"].replace("_", " ")
    name_vi = a.get("name_vi", "")
    desc = a.get("description", "")
    tools = "; ".join(f"{t['name'].split('.')[-1].replace('_', ' ')}: {t.get('description', '')}" for t in a.get("tools", []))
    return f"{name_vi} ({app_id}): {desc}. Các chức năng: {tools}"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class AppRetrievalDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_data(train_folds: set[int], eval_folds: set[int]):
    cv = json.loads((PROJECT / "splits/cv5.json").read_text(encoding="utf-8"))
    catalog = load_catalog("train")
    apps = {x["id"]: x for x in catalog["apps"]}
    app_docs = {aid: enriched_app_doc(a) for aid, a in apps.items()}
    samples = load_samples("train")
    asr_ft = {json.loads(l)["audio"]: json.loads(l)["hyp"]
              for l in (PROJECT / "work/asr/phowhisper_ft/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    train_items, eval_items = [], []
    for s in samples:
        f = cv.get(s["id"], 0)
        gold_apps = {c["name"].split(".")[0] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}
        if not gold_apps:
            continue
        installed = [a for a in s["context"]["installed_apps"] if a in apps]
        neg_apps = [a for a in installed if a not in gold_apps]

        for text_type in ("gold", "asr"):
            q_raw = s["audio_text"] if text_type == "gold" else asr_ft.get(s["audio"], s["audio_text"])
            query = normalize_text(q_raw)

            if f in train_folds:
                for g_app in gold_apps:
                    train_items.append({
                        "query": f"query: {query}",
                        "pos_doc": f"passage: {app_docs[g_app]}",
                        "pos_app": g_app,
                        "neg_docs": [f"passage: {app_docs[na]}" for na in neg_apps],
                        "installed": installed,
                        "gold": list(gold_apps),
                        "sid": s["id"],
                    })
            elif f in eval_folds:
                eval_items.append({
                    "query": f"query: {query}",
                    "text_type": text_type,
                    "sid": s["id"],
                    "installed": installed,
                    "gold": list(gold_apps),
                })
    return train_items, eval_items, app_docs, apps


def evaluate_filter(model, tok, eval_items: list[dict], app_docs: dict[str, str], device: str):
    model.eval()
    app_ids = sorted(app_docs.keys())
    with torch.no_grad():
        doc_texts = [f"passage: {app_docs[aid]}" for aid in app_ids]
        doc_enc = tok(doc_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        doc_emb = F.normalize(mean_pooling(model(**doc_enc), doc_enc["attention_mask"]), p=2, dim=1)
        doc_map = {aid: doc_emb[i] for i, aid in enumerate(app_ids)}

    queries = [item["query"] for item in eval_items]
    q_embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            b_texts = queries[i:i + batch_size]
            b_enc = tok(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            b_emb = F.normalize(mean_pooling(model(**b_enc), b_enc["attention_mask"]), p=2, dim=1)
            q_embs.append(b_emb.cpu())
    q_embs = torch.cat(q_embs, dim=0)

    # Strategy tuning
    thresholds = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    top_ks = [4, 5, 6, 7, 8, 9]

    results = {}
    for text_type in ("gold", "asr"):
        idx_list = [idx for idx, it in enumerate(eval_items) if it["text_type"] == text_type]
        sub_items = [eval_items[idx] for idx in idx_list]
        sub_q = q_embs[idx_list]

        print(f"\n=== Evaluation on Fold 0 ({text_type.upper()} text: {len(sub_items)} samples) ===")
        print("Ranker / Thresholding: Top-K + Threshold (with Core Apps: contacts + sys always kept)")
        print(f"{'K_min':<6} {'Threshold':<10} {'K_max':<6} | {'Sample Recall %':<16} {'Avg Apps Kept':<14} {'Goal (>=99%)'}")
        print("-" * 65)

        best_cfg = None
        for k_min in [3, 4, 5]:
            for th in thresholds:
                for k_max in [k for k in top_ks if k >= k_min]:
                    rec_count = 0
                    kept_counts = []
                    for it, q_vec in zip(sub_items, sub_q):
                        inst = it["installed"]
                        gold = set(it["gold"])
                        # Compute similarity against all installed apps
                        inst_embs = torch.stack([doc_map[a] for a in inst]).cpu()
                        sims = (inst_embs @ q_vec).numpy()
                        ranked_inst = [a for _, a in sorted(zip(sims, inst), key=lambda x: -x[0])]
                        sim_dict = dict(zip(inst, sims))

                        # Apply Strategy A (contacts) + Strategy C (Dynamic Thresholding + Top-K)
                        kept = set(ranked_inst[:k_min])
                        if th > 0.0:
                            above_th = [a for a in ranked_inst if sim_dict[a] >= th]
                            kept.update(above_th[:k_max])
                        if "contacts" in inst:
                            kept.add("contacts")

                        if gold <= kept:
                            rec_count += 1
                        kept_counts.append(len(kept))

                    recall = 100.0 * rec_count / len(sub_items)
                    avg_kept = float(np.mean(kept_counts))
                    pass_goal = "PASS >= 99%" if recall >= 99.0 else "FAIL"

                    if recall >= 98.0 or (k_min == 5 and th == 0.0 and k_max == 5):
                        print(f"{k_min:<6} {th:<10.2f} {k_max:<6} | {recall:6.2f}%         {avg_kept:6.2f} apps     {pass_goal}")

                    if recall >= 99.0:
                        if best_cfg is None or avg_kept < best_cfg["avg_kept"]:
                            best_cfg = {"k_min": k_min, "th": th, "k_max": k_max, "recall": recall, "avg_kept": avg_kept}
        results[text_type] = best_cfg
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test on small subset")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate pre-trained base model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (default: work/models/app_embedder_e5)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    # Prefer local base weights (packed in repo/work/models/app_embedder_e5);
    # fall back to HuggingFace hub when running locally with internet.
    local_base = PROJECT / "work" / "models" / "app_embedder_e5"
    model_src = str(local_base) if (local_base / "config.json").exists() else MODEL_NAME
    tok = AutoTokenizer.from_pretrained(model_src)
    model = AutoModel.from_pretrained(model_src).to(device)
    print(f"  loaded from {model_src}")

    # Fold 0 strictly held out!
    train_folds = {1, 2, 3, 4}
    eval_folds = {0}

    train_items, eval_items, app_docs, apps = build_data(train_folds, eval_folds)
    print(f"Data built: {len(train_items)} train triplets, {len(eval_items)} eval queries on Fold 0.")

    if args.smoke:
        train_items = train_items[:30]
        eval_items = eval_items[:30]
        args.epochs = 1

    if args.eval_only:
        print("\n--- Evaluating Base Model (Zero-Shot) ---")
        evaluate_filter(model, tok, eval_items, app_docs, device)
        return

    # Listwise InfoNCE over the sample's installed apps (the filter's real choice set). In-batch negatives do not work
    # here: only 46 app docs exist, so a batch of 64 holds the same positive doc many times and the in-batch loss pushes
    # a query away from its own app. Each step embeds all 46 docs once; per query the positive is its gold app, the
    # negatives are the other installed apps, and the sample's other gold apps are masked out.
    print(f"\n--- Fine-tuning {MODEL_NAME} (listwise InfoNCE over installed apps) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    app_ids = sorted(app_docs)
    col = {aid: i for i, aid in enumerate(app_ids)}
    d_enc = tok([f"passage: {app_docs[aid]}" for aid in app_ids], padding=True, truncation=True, max_length=256,
                return_tensors="pt").to(device)
    tau = 0.05

    random.seed(42)
    for epoch in range(args.epochs):
        random.shuffle(train_items)
        total_loss, steps = 0.0, 0
        t0 = time.time()
        for i in range(0, len(train_items), args.batch_size):
            batch = train_items[i:i + args.batch_size]
            queries = [b["query"] for b in batch]

            q_enc = tok(queries, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            q_emb = F.normalize(mean_pooling(model(**q_enc), q_enc["attention_mask"]), p=2, dim=1)
            d_emb = F.normalize(mean_pooling(model(**d_enc), d_enc["attention_mask"]), p=2, dim=1)

            sim_matrix = (q_emb @ d_emb.T) / tau
            allowed = torch.zeros_like(sim_matrix, dtype=torch.bool)
            for r, b in enumerate(batch):
                for a in b["installed"]:
                    if a not in b["gold"] or a == b["pos_app"]:
                        allowed[r, col[a]] = True
            sim_matrix = sim_matrix.masked_fill(~allowed, float("-inf"))
            labels = torch.tensor([col[b["pos_app"]] for b in batch], device=device)

            loss = F.cross_entropy(sim_matrix, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps % 25 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {steps} | Loss: {total_loss/steps:.4f} ({time.time()-t0:.1f}s)")

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/max(1, steps):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nSaved fine-tuned embedder to {out_dir}")

    print("\n--- Evaluating Fine-tuned Model on Held-out Fold 0 ---")
    best_results = evaluate_filter(model, tok, eval_items, app_docs, device)
    print("\nOptimal Filter Configurations:", json.dumps(best_results, indent=2))
    if args.out_dir:
        cfg_path = Path(args.out_dir) / "best_config.json"
        cfg_path.write_text(json.dumps(best_results, indent=2), encoding="utf-8")
        print(f"Saved best_config.json -> {cfg_path}")


if __name__ == "__main__":
    main()
