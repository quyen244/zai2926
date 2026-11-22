"""DPO (+ SFT term on the chosen step, RPO-style) of the RAT LoRA on pairs from scripts/dpo_data.py; one GPU.

  python -m lifemate.dpo_train --init-adapter <RAT adapter> --tag rat --out out/traj_dpo
  ... --limit 4 --max-steps 1      # smoke test

Model: Qwen3-4B 4-bit NF4 + fp16 compute, LoRA r16/a32 on every attention and MLP projection, started from the RAT
adapter (lifemate.sft_train.load_adapter), same backends as SFT (unsloth default, hf for a local check).
A pair is two rows that differ only in the last assistant message: the prompt + gold prefix, then chosen (gold step)
or rejected (RAT's own step). Each is rendered and masked exactly as SFT (lifemate.sft.encode: loss on that message
only); the token prefix before the supervised span must be identical in both, or the pair is refused.
logp(y) = sum of token log-probs over the supervised span. The reference is the initial RAT adapter itself: every pair's
reference logps are computed once before the first step (no second model in memory).
Loss per pair: -log sigmoid(beta * ((pc - rc) - (pr - rr))) + alpha * mean NLL of the chosen tokens.
Memory: the two sequences never hold a graph at the same time. pr is first computed without grad, the chosen branch is
back-propagated with pr fixed (its exact gradient), then the rejected branch is re-run with grad and back-propagated
with the weight d loss / d pr = beta * sigmoid(-z) (exact as well; z from the same weights, nothing changed in between).
Writes <out>/adapter, <out>/log.jsonl (per optimizer step), <out>/stats.json (val preference accuracy and margin before
and after, reward accuracy over training, time, memory).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .sft import IGNORE, encode
from .sft_train import load_adapter, t4_attention

ROOT = Path(__file__).resolve().parent.parent


def load_pairs(data_dir: str, part: str, tag: str = "rec") -> list[dict]:
    p = ROOT / data_dir / f"{tag}.{part}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode_pair(r: dict, tok, max_len: int) -> dict:
    out = {}
    for side in ("chosen", "rejected"):
        e = encode({**r, "messages": r["messages"] + [{**r[side], "train": True}]}, tok, max_len)
        out[side] = (e["input_ids"], e["labels"])
    (ci, cl), (ri, rl) = out["chosen"], out["rejected"]
    first = lambda lab: next(k for k, x in enumerate(lab) if x != IGNORE)
    fc, fr = first(cl), first(rl)
    if fc != fr or ci[:fc] != ri[:fr]:
        raise ValueError(f"{r['id']} {r['variant']}: chosen and rejected prompts differ ({fc} vs {fr} tokens)")
    return {"id": r["id"], "variant": r["variant"], "kind": r["kind"], "chosen": out["chosen"], "rejected": out["rejected"],
            "n_tokens": max(len(ci), len(ri))}


def seq_logp(lm, ids: list[int], labels: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum of log p over the supervised tokens, mean NLL over them); lm_head runs on those positions only."""
    dev = next(lm.parameters()).device
    x = torch.tensor([ids], device=dev)
    y = torch.tensor([labels], device=dev)
    h = lm.model(input_ids=x, use_cache=False).last_hidden_state
    tgt = y[:, 1:]
    keep = tgt != IGNORE
    logits = lm.lm_head(h[:, :-1][keep]).float()
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[keep][:, None]).squeeze(-1)
    return lp.sum(), -lp.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", default="work/dpo")
    ap.add_argument("--tag", default="rec", help="pair file prefix, e.g. rat for rat.train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-len", type=int, default=10240)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the chosen-step NLL (0 = plain DPO)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the N longest train pairs")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this many hours of training, then evaluate and save")
    ap.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    a = ap.parse_args()

    if a.backend == "unsloth":
        import unsloth  # noqa: F401
    from transformers import AutoTokenizer, set_seed

    set_seed(a.seed)
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    train = [encode_pair(r, tok, a.max_len) for r in load_pairs(a.data_dir, "train", a.tag)]
    val = [encode_pair(r, tok, a.max_len) for r in load_pairs(a.data_dir, "val", a.tag)]
    if a.limit:
        train = sorted(train, key=lambda x: -x["n_tokens"])[: a.limit]
        val = val[: min(len(val), 4)]
    print(f"encoded {len(train)} train / {len(val)} val pairs in {time.time() - t0:.0f}s; max tokens {max(x['n_tokens'] for x in train)}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if a.backend == "unsloth":
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
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    load_adapter(model, a.init_adapter, a.rank, targets)
    model.print_trainable_parameters()
    lm = model.get_base_model()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # fp32 master weights for the LoRA matrices (fp16 grads would underflow at lr 1e-5)
        p.data = p.data.float()
    amp = lambda: torch.autocast("cuda", dtype=torch.float16)
    attn = t4_attention() if a.backend == "hf" else contextlib.nullcontext()

    def logp(x: dict, side: str, grad: bool):
        with (contextlib.nullcontext() if grad else torch.no_grad()), amp():
            return seq_logp(lm, *x[side])

    def evaluate(pairs: list[dict], ref: dict) -> dict:
        model.eval()
        acc, margin = 0, 0.0
        for k, x in enumerate(pairs):
            pc, pr = logp(x, "chosen", False)[0].item(), logp(x, "rejected", False)[0].item()
            rc, rr = ref[k]
            m = (pc - rc) - (pr - rr)
            margin += m
            acc += (pc - pr) > 0  # the policy itself prefers the chosen step
        model.train()
        return {"pref_acc": acc / max(len(pairs), 1), "dpo_margin": margin / max(len(pairs), 1)}

    with attn:
        t_ref = time.time()
        model.eval()
        ref_train = [(logp(x, "chosen", False)[0].item(), logp(x, "rejected", False)[0].item()) for x in train]
        ref_val = [(logp(x, "chosen", False)[0].item(), logp(x, "rejected", False)[0].item()) for x in val]
        model.train()
        start = {"pref_acc": sum(c > r for c, r in ref_val) / max(len(ref_val), 1), "dpo_margin": 0.0}
        print(f"reference logps for {len(train) + len(val)} pairs in {time.time() - t_ref:.0f}s; initial RAT val pref acc {start['pref_acc']:.3f}", flush=True)

        order = []
        n_ep = math.ceil(a.epochs)
        for e in range(n_ep):
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx[: round(len(idx) * min(1.0, a.epochs - e))]
        steps_total = math.ceil(len(order) / a.accum) if a.max_steps < 0 else a.max_steps
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr, weight_decay=0.0)
        warm = max(1, round(0.1 * steps_total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
        scaler = torch.amp.GradScaler("cuda")
        log = open(a.out / "log.jsonl", "w", encoding="utf-8")
        torch.cuda.reset_peak_memory_stats()
        t_train, step, seen, stopped = time.time(), 0, 0, False
        acc_win, loss_win, marg_win = [], [], []
        evals = []
        for start_i in range(0, len(order), a.accum):
            if step >= steps_total:
                break
            for k in order[start_i: start_i + a.accum]:
                x, (rc, rr) = train[k], ref_train[k]
                pr0 = logp(x, "rejected", False)[0]
                pc, nll = logp(x, "chosen", True)
                z = a.beta * ((pc - rc) - (pr0 - rr))
                loss_c = -F.logsigmoid(z) + a.alpha * nll
                scaler.scale(loss_c / a.accum).backward()
                w = (a.beta * torch.sigmoid(-z)).detach()
                pr, _ = logp(x, "rejected", True)
                scaler.scale(w * pr / a.accum).backward()
                acc_win.append(float(z.item() > 0))
                loss_win.append((-F.logsigmoid(z)).item())
                marg_win.append(z.item() / a.beta)
                seen += 1
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            rec = {"step": step, "pairs": seen, "time_s": round(time.time() - t_train), "lr": sched.get_last_lr()[0], "grad_norm": round(gn, 3),
                   "dpo_loss": round(sum(loss_win) / len(loss_win), 4), "reward_acc": round(sum(acc_win) / len(acc_win), 3),
                   "margin": round(sum(marg_win) / len(marg_win), 3), "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
            acc_win, loss_win, marg_win = [], [], []
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if step % max(1, steps_total // 2) == 0 and step < steps_total:
                ev = {"step": step, **evaluate(val, ref_val)}
                evals.append(ev)
                print("val", json.dumps(ev), flush=True)
            if a.max_hours and time.time() - t_train > a.max_hours * 3600:
                stopped = True
                print(f"time limit {a.max_hours} h reached at step {step}: stopping", flush=True)
                break
        final = evaluate(val, ref_val)
    wall = time.time() - t_train
    model.save_pretrained(a.out / "adapter")
    tok.save_pretrained(a.out / "adapter")
    stats = {"pairs_train": len(train), "pairs_val": len(val), "steps": step, "steps_planned": steps_total, "pairs_seen": seen,
             "stopped_by_time_limit": stopped, "val_start": start, "val_mid": evals, "val_final": final,
             "train_wall_s": round(wall), "gpu_hours": round(wall / 3600, 2), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
             "gpu": torch.cuda.get_device_name(0), "backend": a.backend, "hparams": vars(a) | {"out": str(a.out), "init_adapter": str(a.init_adapter)}}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    print("done:", json.dumps(stats, default=str), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
