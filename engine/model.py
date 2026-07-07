# Vendored from pytorch-labs/gpt-fast model.py (commit in GPTFAST_COMMIT.txt), adapted:
#  - SDPA attention with a dense causal-mask buffer instead of flex_attention/BlockMask
#    (stable on torch>=2.5, CUDA-graph friendly; the one structural deviation from upstream HEAD)
#  - Qwen2 support: qkv biases (attn_bias), tied embeddings flag, Qwen2.5 configs
# Original: Copyright (c) Meta Platforms, Inc. and affiliates. BSD-licensed.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    attn_bias: bool = False           # Qwen2: q/k/v projections have biases
    tie_word_embeddings: bool = False  # Qwen2.5-0.5B: output = tok_embeddings

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [c for c in transformer_configs if c.lower() in str(name).lower()]
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name
        return cls(**transformer_configs[config[0]])


# Qwen2.5 values asserted against each model's HF config.json by convert_qwen.py.
transformer_configs = {
    "Qwen2.5-0.5B-Instruct": dict(
        block_size=4096, vocab_size=151936, n_layer=24, n_head=14, n_local_heads=2,
        dim=896, intermediate_size=4864, rope_base=1000000.0, norm_eps=1e-6,
        attn_bias=True, tie_word_embeddings=True),
    "Qwen2.5-14B-Instruct": dict(
        block_size=4096, vocab_size=152064, n_layer=48, n_head=40, n_local_heads=8,
        dim=5120, intermediate_size=13824, rope_base=1000000.0, norm_eps=1e-6,
        attn_bias=True),
    "Qwen2.5-32B-Instruct": dict(
        block_size=4096, vocab_size=152064, n_layer=64, n_head=40, n_local_heads=8,
        dim=5120, intermediate_size=27648, rope_base=1000000.0, norm_eps=1e-6,
        attn_bias=True),
}


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.causal_mask: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)
        # fp32 rope cache regardless of model dtype: HF applies rope in fp32, and bf16 angle
        # quantization was measured to cost ~0.9% argmax agreement vs HF (apply_rotary_emb
        # already computes in fp32 and casts back, so this is dtype-compatible).
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size, head_dim, self.config.rope_base, torch.float32)
        self.causal_mask = torch.tril(
            torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch (Qwen2: with bias)
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.attn_bias)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor,
                input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # GQA via zero-copy batch-fold (batch=1 only): enable_gqa+mask has no efficient kernel
        # in torch 2.5 (math fallback, 161us/layer) and repeat_interleave materializes 5x KV
        # (109us); folding kv-groups into batch with a stride-0 expand hits the cutlass kernel
        # with no copy (68us, bit-exact).
        if self.n_head != self.n_local_heads and bsz == 1:
            rep = self.n_head // self.n_local_heads
            L = q.shape[2]
            S = k.shape[2]
            qg = q.reshape(self.n_local_heads, rep, L, self.head_dim)   # q transposed -> reshape
            kg = k.view(self.n_local_heads, 1, S, self.head_dim).expand(-1, rep, -1, -1)
            vg = v.view(self.n_local_heads, 1, S, self.head_dim).expand(-1, rep, -1, -1)
            y = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask)
            y = y.reshape(bsz, self.n_head, L, self.head_dim)           # sdpa out may be non-contig
        else:
            if self.n_head != self.n_local_heads:
                rep = self.n_head // self.n_local_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000,
                         dtype: torch.dtype = torch.bfloat16) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
