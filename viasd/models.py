"""Model loading, layer-skipped slim-verifier (q'), and latency measurement."""
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .cost import Latencies


@dataclass
class Tiers:
    drafter: nn.Module
    verifier: nn.Module
    tokenizer: object
    keep_mask: list           # bool per verifier layer; True = keep, False = skip (q')
    vocab: int                # common vocab = min(drafter, verifier); trailing slots are padding
    cfg: Config

    @property
    def device(self):
        return self.cfg.device


def _decoder_layers(model) -> nn.ModuleList:
    """Locate the ModuleList of transformer blocks (Qwen2/Llama-style)."""
    return model.model.layers


@contextmanager
def skipped_layers(model, keep_mask):
    """Temporarily run only the kept layers. Each decoder block is residual
    (x = x + sublayer(x)), so dropping a block is an identity on the residual
    stream -- dimensionally safe. Forwards must use use_cache=False so the
    (now non-contiguous) per-layer cache indices are never used."""
    layers = _decoder_layers(model)
    full = layers
    kept = nn.ModuleList([l for l, k in zip(layers, keep_mask) if k])
    model.model.layers = kept
    try:
        yield
    finally:
        model.model.layers = full


def make_keep_mask(n_layers, skip_ratio, keep_first_last=True):
    """Skip `skip_ratio` of layers, evenly spaced over the middle."""
    n_skip = int(round(skip_ratio * n_layers))
    keep = [True] * n_layers
    candidates = list(range(1, n_layers - 1)) if keep_first_last else list(range(n_layers))
    n_skip = min(n_skip, len(candidates))
    if n_skip > 0:
        # evenly spaced indices among candidates
        step = len(candidates) / n_skip
        drop = sorted({candidates[min(len(candidates) - 1, int(i * step))] for i in range(n_skip)})
        for d in drop:
            keep[d] = False
    return keep


def load_models(cfg: Config) -> Tiers:
    tok = AutoTokenizer.from_pretrained(cfg.verifier_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = cfg.torch_dtype()
    common = dict(dtype=dtype, attn_implementation="sdpa")
    drafter = AutoModelForCausalLM.from_pretrained(cfg.drafter_name, **common).to(cfg.device).eval()
    verifier = AutoModelForCausalLM.from_pretrained(cfg.verifier_name, **common).to(cfg.device).eval()
    for m in (drafter, verifier):
        for p in m.parameters():
            p.requires_grad_(False)

    n_layers = len(_decoder_layers(verifier))
    if cfg.keep_mask_path:
        import json
        with open(cfg.keep_mask_path) as f:
            keep_mask = json.load(f)["keep_mask"]
        assert len(keep_mask) == n_layers, "keep_mask length mismatch with verifier"
    else:
        keep_mask = make_keep_mask(n_layers, cfg.skip_ratio, cfg.keep_first_last)
    vocab = min(drafter.config.vocab_size, verifier.config.vocab_size)
    return Tiers(drafter, verifier, tok, keep_mask, vocab, cfg)


@torch.no_grad()
def lm_logits(model, input_ids, keep_mask=None):
    """Full-stack (or layer-skipped) logits over all positions, no cache."""
    if keep_mask is not None:
        with skipped_layers(model, keep_mask):
            return model(input_ids=input_ids, use_cache=False).logits
    return model(input_ids=input_ids, use_cache=False).logits


def compile_models(tiers: Tiers, mode="default"):
    """torch.compile the drafter and verifier to cut per-call overhead.
    The layer-skip context manager mutates verifier.model.layers, which triggers
    a recompile when switching between full/slim graphs -- fine as long as the two
    shapes are each reused (grouped) rather than interleaved per call."""
    tiers.drafter = torch.compile(tiers.drafter, mode=mode)
    tiers.verifier = torch.compile(tiers.verifier, mode=mode)
    return tiers


@torch.no_grad()
def measure_latencies(tiers: Tiers, ctx_len=256, gamma=None, reps=8, warmup=3) -> Latencies:
    """Time one forward of each tier on the real device. q'/q are timed as
    gamma-token parallel block forwards; drafter and baseline-q as 1-token
    cached steps. Returns seconds/forward (median-ish via mean of `reps`)."""
    cfg = tiers.cfg
    gamma = gamma or cfg.gamma
    dev = cfg.device
    V = tiers.vocab
    ctx = torch.randint(0, V, (1, ctx_len), device=dev)
    blk = torch.randint(0, V, (1, ctx_len + gamma), device=dev)
    one = torch.randint(0, V, (1, 1), device=dev)

    def timeit(fn):
        for _ in range(warmup):  # warmup (also triggers torch.compile for this shape)
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    # drafter cached 1-token step
    out = tiers.drafter(input_ids=ctx, use_cache=True)
    past = out.past_key_values
    t_p1 = timeit(lambda: tiers.drafter(input_ids=one, past_key_values=past, use_cache=True))

    # full verifier cached 1-token step (baseline autoregressive)
    outq = tiers.verifier(input_ids=ctx, use_cache=True)
    pastq = outq.past_key_values
    t_q1 = timeit(lambda: tiers.verifier(input_ids=one, past_key_values=pastq, use_cache=True))

    # parallel block forwards
    t_q = timeit(lambda: lm_logits(tiers.verifier, blk, None))
    t_qp = timeit(lambda: lm_logits(tiers.verifier, blk, tiers.keep_mask))
    return Latencies(t_p1=t_p1, t_qp=t_qp, t_q=t_q, t_q1=t_q1)
