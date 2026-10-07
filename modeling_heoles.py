import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

TF_MAJOR = int(transformers.__version__.split(".")[0])


class HeolesConfig(PretrainedConfig):
    model_type = "heoles"

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=8192,
        sliding_window=1024,
        global_every=6,
        rms_norm_eps=1e-6,
        rope_theta_local=10000.0,
        rope_theta_global=1000000.0,
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
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.global_every = global_every
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def is_global(self, layer_idx):
        return (layer_idx + 1) % self.global_every == 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


class KVCache:
    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers
        self.seen = 0

    def stored(self, layer_idx):
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[2]


def rope_cos_sin(positions, dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
    freqs = torch.outer(positions.float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def band_mask(t, offset, k_len, k_start, window, device):
    q = torch.arange(offset, offset + t, device=device)[:, None]
    k = torch.arange(k_start, k_start + k_len, device=device)[None, :]
    m = k <= q
    if window is not None:
        m = m & ((q - k) < window)
    return m


class Attention(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = c.is_global(layer_idx)
        self.window = None if self.is_global else c.sliding_window
        self.nh = c.num_attention_heads
        self.nkv = c.num_key_value_heads
        self.hd = c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, c.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, c.rms_norm_eps)

    def forward(self, x, cos, sin, mask, causal, cache):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if cache is not None:
            if cache.k[self.layer_idx] is not None:
                k = torch.cat([cache.k[self.layer_idx], k], dim=2)
                v = torch.cat([cache.v[self.layer_idx], v], dim=2)
            keep = k if self.window is None else k[:, :, -(self.window - 1):]
            keep_v = v if self.window is None else v[:, :, -(self.window - 1):]
            cache.k[self.layer_idx] = keep
            cache.v[self.layer_idx] = keep_v
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        return self.o_proj(o.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.attn = Attention(c, layer_idx)
        self.mlp = MLP(c)
        self.pre_attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, cos, sin, mask, causal, cache):
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), cos, sin, mask, causal, cache))
        x = x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))
        return x


class HeolesForCausalLM(PreTrainedModel):
    config_class = HeolesConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"} if TF_MAJOR >= 5 else ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.use_checkpointing = False
        self.post_init()
        scale = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def enable_checkpointing(self, flag=True):
        self.use_checkpointing = flag

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, **kwargs):
        c = self.config
        b, t = input_ids.shape
        device = input_ids.device
        cache = past_key_values
        if use_cache and cache is None:
            cache = KVCache(c.num_hidden_layers)
        offset = cache.seen if cache is not None else 0

        x = self.embed_tokens(input_ids) * math.sqrt(c.hidden_size)

        positions = torch.arange(offset, offset + t, device=device)
        cos_l, sin_l = rope_cos_sin(positions, c.head_dim, c.rope_theta_local)
        cos_g, sin_g = rope_cos_sin(positions, c.head_dim, c.rope_theta_global)

        first_local = next((i for i in range(c.num_hidden_layers) if not c.is_global(i)), None)
        local_stored = cache.stored(first_local) if (cache is not None and first_local is not None) else 0

        if t == 1:
            mask_g = mask_l = None
            causal_g = causal_l = False
        else:
            if offset == 0:
                mask_g, causal_g = None, True
                if t <= c.sliding_window:
                    mask_l, causal_l = None, True
                else:
                    mask_l = band_mask(t, 0, t, 0, c.sliding_window, device)
                    causal_l = False
            else:
                mask_g = band_mask(t, offset, offset + t, 0, None, device)
                causal_g = False
                mask_l = band_mask(t, offset, local_stored + t, offset - local_stored, c.sliding_window, device)
                causal_l = False

        for i, layer in enumerate(self.layers):
            g = c.is_global(i)
            cos, sin = (cos_g, sin_g) if g else (cos_l, sin_l)
            mask, causal = (mask_g, causal_g) if g else (mask_l, causal_l)
            if self.use_checkpointing and self.training and cache is None:
                x = checkpoint(
                    lambda h, l=layer, co=cos, si=sin, m=mask, ca=causal: l(h, co, si, m, ca, None),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin, mask, causal, cache)

        if cache is not None:
            cache.seen = offset + t

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.1,
        stop_ids=None,
        streamer=None,
        **kwargs,
    ):
        self.eval()
        stop = set(stop_ids if stop_ids is not None else [self.config.eos_token_id])
        out = self(input_ids, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
        seq = input_ids
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, seq)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits = logits.scatter(1, seq, score)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if min_p > 0.0:
                    probs = probs.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, 0.0)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = sp.cumsum(-1)
                    sp = sp.masked_fill(cum - sp > top_p * sp.sum(-1, keepdim=True), 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=-1)
            if streamer is not None:
                streamer(nxt)
            if nxt.shape[0] == 1 and nxt.item() in stop:
                break
            logits = self(nxt, past_key_values=cache, use_cache=True).logits[:, -1, :].float()
        return seq
