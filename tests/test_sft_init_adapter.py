"""lifemate.sft_train.load_adapter: a saved LoRA adapter is copied exactly into a fresh one, mismatches fail loudly.
Tiny random Qwen3 (2 layers, CPU); no real model is loaded."""
import pytest
import torch

peft = pytest.importorskip("peft")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from lifemate.sft_train import load_adapter  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny(rank: int = 4, targets=TARGETS):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    return peft.get_peft_model(Qwen3ForCausalLM(cfg), peft.LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets, lora_dropout=0))


def lora(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def test_loads_exactly_and_stays_trainable(tmp_path):
    src = tiny()
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "lora_B" in n:
                p.normal_()
    src.save_pretrained(tmp_path)
    dst = tiny()
    assert any(not torch.equal(lora(dst)[k], v) for k, v in lora(src).items())
    load_adapter(dst, tmp_path, 4, TARGETS)
    got, want = lora(dst), lora(src)
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in want)
    assert all(p.requires_grad for n, p in dst.named_parameters() if "lora_" in n)


def test_rank_mismatch_fails(tmp_path):
    tiny(rank=4).save_pretrained(tmp_path)
    with pytest.raises(SystemExit):
        load_adapter(tiny(rank=8), tmp_path, 8, TARGETS)


def test_fresh_adapter_refused(tmp_path):
    tiny().save_pretrained(tmp_path)  # lora_B is zero at init: loading it would silently restart from scratch
    with pytest.raises(SystemExit):
        load_adapter(tiny(), tmp_path, 4, TARGETS)
