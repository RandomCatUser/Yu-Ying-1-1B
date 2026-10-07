import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class HeolesConfig(PretrainedConfig):
    model_type = "heoles"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(seq_len, dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c.num_attention_heads
        self.nkv = c.num_key_value_heads
        self.hd = c.hidden_size // c.num_attention_heads
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn = Attention(c)
        self.mlp_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class HeolesForCausalLM(PreTrainedModel):
    config_class = HeolesConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, labels=None, **kwargs):
        c = self.config
        x = self.embed_tokens(input_ids)
        cos, sin = rope_cache(input_ids.shape[1], c.hidden_size // c.num_attention_heads, c.rope_theta, input_ids.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.8, top_k=50, top_p=0.95, **kwargs):
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.max_position_embeddings:]
            logits = self(window).logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = sp.cumsum(-1)
                    sp = sp.masked_fill(cum - sp > top_p, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=-1)
            if (nxt == self.config.eos_token_id).all():
                break
        return input_ids
