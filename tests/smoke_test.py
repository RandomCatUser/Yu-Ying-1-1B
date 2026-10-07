import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_yuying import YuYingConfig, YuYingForCausalLM


def small():
    cfg = YuYingConfig(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        global_every=3,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return YuYingForCausalLM(cfg).eval()


def test_cache_matches_full():
    m = small()
    ids = torch.randint(3, 500, (2, 40))
    full = m(ids).logits
    out = m(ids[:, :25], use_cache=True)
    cache = out.past_key_values
    parts = [out.logits]
    for s in range(25, 40, 5):
        parts.append(m(ids[:, s : s + 5], past_key_values=cache, use_cache=True).logits)
    assert (full - torch.cat(parts, dim=1)).abs().max().item() < 1e-4
    out = m(ids[:, :20], use_cache=True)
    cache = out.past_key_values
    parts = [out.logits]
    for s in range(20, 40):
        parts.append(m(ids[:, s : s + 1], past_key_values=cache, use_cache=True).logits)
    assert (full - torch.cat(parts, dim=1)).abs().max().item() < 1e-4


def test_checkpointing_gradients():
    m = small().train()
    ids = torch.randint(3, 500, (2, 40))
    m.enable_checkpointing(True)
    m(ids, labels=ids).loss.backward()
    a = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.enable_checkpointing(False)
    m(ids, labels=ids).loss.backward()
    b = [p.grad for p in m.parameters()]
    assert max((x - y).abs().max().item() for x, y in zip(a, b)) < 1e-5


def test_greedy_generate_matches_no_cache():
    m = small()
    prompt = torch.randint(3, 500, (1, 5))
    got = m.generate(prompt, max_new_tokens=12, temperature=0, repetition_penalty=1.0, stop_ids=[-1])
    seq = prompt
    for _ in range(12):
        seq = torch.cat([seq, m(seq).logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(got, seq)


def test_save_and_reload():
    m = small()
    ids = torch.randint(3, 500, (1, 16))
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = YuYingForCausalLM.from_pretrained(d).eval()
    assert (m(ids).logits - m2(ids).logits).abs().max().item() < 1e-6
    assert m2.lm_head.weight.data_ptr() == m2.embed_tokens.weight.data_ptr()


def test_full_size_is_about_one_billion():
    with torch.device("meta"):
        big = YuYingForCausalLM(YuYingConfig())
    n = sum(p.numel() for p in big.parameters())
    assert 0.95e9 < n < 1.05e9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
