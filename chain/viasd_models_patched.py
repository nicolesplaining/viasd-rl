"""Model loading, layer-skipped slim-verifier (q'), and latency measurement."""
import os
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
    draft_mask: list = None    # whole-block keep_mask for a SELF-SPEC draft (KnapSpec-tuned); None = use the separate drafter

    @property
    def device(self):
        return self.cfg.device


def _decoder_layers(model) -> nn.ModuleList:
    """Locate the ModuleList of transformer blocks (Qwen2/Llama-style)."""
    return model.model.layers


def _draft_active_params(tiers):
    """Active params of the DRAFT tier: the separate drafter, or (self-spec) the verifier
    with only its kept whole-blocks."""
    if getattr(tiers, "draft_mask", None) is None:
        return sum(p.numel() for p in tiers.drafter.parameters())
    layers = _decoder_layers(tiers.verifier)
    lp = sum(p.numel() for l in layers for p in l.parameters())
    total = sum(p.numel() for p in tiers.verifier.parameters())
    return (total - lp) + sum(tiers.draft_mask) * (lp / len(layers))


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


class _ZeroAttn(nn.Module):
    """Drop-in for a decoder layer's self_attn: returns a zero contribution so the
    residual stream is unchanged (x = x + 0). The real attn matmuls never run, so
    both compute (incl. O(n^2) attention) and the attn weight-read are saved."""
    def forward(self, hidden_states=None, *a, **k):
        hs = hidden_states if hidden_states is not None else (a[0] if a else k.get("hidden_states"))
        return torch.zeros_like(hs), None   # layer does: hidden_states, _ = self_attn(...)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


@contextmanager
def skipped_sublayers(model, skip_set):
    """Sublayer (attn/MLP-decoupled) skipping for q'. skip_set has length 2L:
    [attn_0, mlp_0, attn_1, mlp_1, ...], 1=skip. Each sublayer is residual
    (x = x + sublayer(x)); skipping replaces it with a zero-shim so its matmuls
    and weight-read are elided. This is the KnapSpec-DP granularity: it can drop
    the expensive attention (O(n^2), grows with context) where DIMR can only drop
    whole blocks. use_cache=False (same non-contiguous-cache caveat as skipped_layers)."""
    layers = _decoder_layers(model)
    saved = []
    for i, l in enumerate(layers):
        saved.append((l.self_attn, l.mlp))
        if skip_set[2 * i]:
            l.self_attn = _ZeroAttn()
        if skip_set[2 * i + 1]:
            l.mlp = _ZeroMLP()
    try:
        yield
    finally:
        for l, (a, m) in zip(layers, saved):
            l.self_attn, l.mlp = a, m


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
        assert len(keep_mask) in (n_layers, 2 * n_layers), "mask must be len L (block keep_mask) or 2L (sublayer skip_set)"
    else:
        keep_mask = make_keep_mask(n_layers, cfg.skip_ratio, cfg.keep_first_last)
    vocab = min(drafter.config.vocab_size, verifier.config.vocab_size)
    draft_mask = None
    dmp = os.environ.get("VIASD_DRAFT_MASK")
    if dmp:
        import json
        with open(dmp) as f:
            draft_mask = json.load(f)["keep_mask"]
        assert len(draft_mask) == n_layers, "draft_mask must be a whole-block keep_mask (len L)"
        print(f"[SELF-SPEC DRAFT] verifier with keep {sum(draft_mask)}/{n_layers} layers "
              f"(KnapSpec-tuned draft); separate {cfg.drafter_name} loaded but UNUSED", flush=True)
    else:
        print(f"[DRAFT] separate model: {cfg.drafter_name}", flush=True)
    return Tiers(drafter, verifier, tok, keep_mask, vocab, cfg, draft_mask)


@torch.no_grad()
def lm_logits(model, input_ids, keep_mask=None):
    """Full-stack (or layer-skipped) logits over all positions, no cache.
    Dispatches by mask length: len==n_layers -> whole-block keep_mask (True=keep,
    DIMR/evenly); len==2*n_layers -> sublayer skip_set (1=skip, KnapSpec-DP)."""
    if keep_mask is not None:
        n = len(_decoder_layers(model))
        ctx = skipped_sublayers(model, keep_mask) if len(keep_mask) == 2 * n else skipped_layers(model, keep_mask)
        with ctx:
            return model(input_ids=input_ids, use_cache=False).logits
    return model(input_ids=input_ids, use_cache=False).logits


def compile_models(tiers: Tiers, mode="default", compile_verifier=False):
    """torch.compile the drafter (and optionally verifier) to cut per-call overhead.

    CUDA graphs (mode='reduce-overhead', and inductor's default cudagraph_trees)
    are INCOMPATIBLE with our reused dynamic KV-cache loop -- they raise
    "accessing tensor output of CUDAGraphs that has been overwritten". So we
    disable cudagraphs and rely on inductor kernel *fusion* only (fewer launches,
    no graph capture). This is the safe win; the bigger cudagraph win needs a
    static-cache rewrite.

    The layer-skip context manager mutates verifier.model.layers, which triggers a
    recompile when switching full<->slim graphs -- fine as long as the two shapes
    are each reused (grouped) rather than interleaved per call."""
    try:
        import torch._inductor.config as ic
        ic.triton.cudagraphs = False
    except Exception:
        pass
    # Drafter only: it has no layer-skip, so compiling it is safe and high-value
    # (drafting = gamma sequential steps/block, the dominant per-block latency;
    # measured ~1.8x). The verifier is NOT compiled here because via_sd alternates
    # full<->slim (layer-skip swap) every block, which thrashes torch.compile guards;
    # and the slim path gets ~no benefit anyway (measured 1.00x).
    if compile_verifier:
        tiers.verifier = torch.compile(tiers.verifier)
    tiers.drafter = torch.compile(tiers.drafter)
    return tiers


def bandwidth_latencies(tiers: Tiers, bandwidth_bytes_s=3.35e12, bytes_per_param=2) -> Latencies:
    """Idealized per-forward latency from the batch-1 decode bottleneck: memory
    bandwidth. One decode step streams all (active) weights once, so
    time ~= active_params * bytes_per_param / HBM_bandwidth -- independent of the
    eager per-call overhead floor that pollutes measured wall-clock.

    A gamma-token block verify still reads weights once (compute is negligible vs
    the weight read at batch 1), so t_q == t_q1 and t_qp scales with kept layers.
    Default bandwidth = H100 SXM (~3.35 TB/s); the speedup *ratio* is bandwidth-
    independent. This is "seconds/decode in an overhead-free implementation."
    """
    layers = _decoder_layers(tiers.verifier)
    layer_params = sum(p.numel() for l in layers for p in l.parameters())
    total_q = sum(p.numel() for p in tiers.verifier.parameters())
    non_layer = total_q - layer_params
    per_layer = layer_params / len(layers)
    km = tiers.keep_mask
    if len(km) == 2 * len(layers):
        # sublayer skip_set (1=skip): count attn/mlp params kept per layer (layernorms always run)
        P_qp = non_layer
        for i, l in enumerate(layers):
            attn_p = sum(p.numel() for p in l.self_attn.parameters())
            mlp_p = sum(p.numel() for p in l.mlp.parameters())
            other_p = sum(p.numel() for p in l.parameters()) - attn_p - mlp_p
            P_qp += other_p
            if not km[2 * i]:
                P_qp += attn_p
            if not km[2 * i + 1]:
                P_qp += mlp_p
    else:
        P_qp = non_layer + sum(km) * per_layer   # whole-block keep_mask (True=keep)
    P_p = _draft_active_params(tiers)

    def t(params):
        return params * bytes_per_param / bandwidth_bytes_s

    return Latencies(t_p1=t(P_p), t_qp=t(P_qp), t_q=t(total_q), t_q1=t(total_q))


@torch.no_grad()
def corrected_latencies(tiers: Tiers, raw: "Latencies" = None) -> "Latencies":
    """Overhead-corrected per-forward latency = measured - launch_floor.

    A batch-1 forward costs `floor + params*2/BW_eff` (per-token compute is
    negligible). We solve for the fixed launch `floor` and effective bandwidth
    `BW_eff` from the drafter vs verifier 1-token times (overhead is size-
    independent; weight-read scales with params), then subtract the floor from
    every measured forward. Real measurement, with only the launch artifact removed.
    """
    raw = raw or measure_latencies(tiers)
    P_p = _draft_active_params(tiers)
    P_q = sum(p.numel() for p in tiers.verifier.parameters())
    if raw.t_q1 <= raw.t_p1 or P_q <= P_p:        # degenerate; nothing to correct
        return raw
    bw_eff = (P_q - P_p) * 2 / (raw.t_q1 - raw.t_p1)   # bytes/sec
    floor = max(0.0, raw.t_p1 - P_p * 2 / bw_eff)
    sub = lambda t: max(t - floor, 1e-5)
    return Latencies(t_p1=sub(raw.t_p1), t_qp=sub(raw.t_qp), t_q=sub(raw.t_q), t_q1=sub(raw.t_q1))


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

    # draft cached 1-token step (self-spec: verifier under the draft skip-mask; else the separate drafter)
    if tiers.draft_mask is not None:
        with skipped_layers(tiers.verifier, tiers.draft_mask):
            out = tiers.verifier(input_ids=ctx, use_cache=True)
            past = out.past_key_values
            t_p1 = timeit(lambda: tiers.verifier(input_ids=one, past_key_values=past, use_cache=True))
    else:
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
