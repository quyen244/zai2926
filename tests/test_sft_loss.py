"""selective_loss (lifemate/sft_train.py) must equal the model's own causal-LM loss: same masking, same shift, same
normalisation (sum / num_items_in_batch when given, mean otherwise). Tiny random Qwen3 on CPU."""
import torch

from lifemate.sft import IGNORE
from lifemate.sft_train import selective_loss


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).eval()


def batch():
    torch.manual_seed(1)
    ids = torch.randint(0, 97, (1, 20))
    labels = torch.full_like(ids, IGNORE)
    labels[0, 8:12] = ids[0, 8:12]  # two supervised spans, like two trained assistant turns
    labels[0, 15:19] = ids[0, 15:19]
    return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}


def test_mean_matches_model_loss():
    lm, b = tiny_qwen3(), batch()
    with torch.no_grad():
        ref = lm(**b).loss
        got = selective_loss(lm, b)
    assert torch.allclose(got, ref, atol=1e-5), (got, ref)


def test_sum_over_num_items_matches_model_loss():
    lm, b = tiny_qwen3(), batch()
    n = torch.tensor(13)  # e.g. supervised tokens of the whole accumulated batch
    with torch.no_grad():
        ref = lm(**b, num_items_in_batch=n).loss
        got = selective_loss(lm, b, n)
    assert torch.allclose(got, ref, atol=1e-5), (got, ref)


def test_gradients_match():
    lm, b = tiny_qwen3(), batch()
    lm(**b).loss.backward()
    ref = [p.grad.clone() for p in lm.parameters()]
    lm.zero_grad()
    selective_loss(lm, b).backward()
    for g, r in zip((p.grad for p in lm.parameters()), ref):
        assert torch.allclose(g, r, atol=1e-5)
