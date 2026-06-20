#!/usr/bin/env python3
# latency_sweep.py — Exp 2: q' forward latency vs context length for {full-q, DIMR-block,
# KnapSpec-style sublayer (attn-skip)}. The decisive long-context test: attention is O(n^2),
# so a sublayer q' that drops attention should scale far better than DIMR (block-only, keeps
# attention in every kept block). Pure latency (measured), no task run -> fast.
import os, sys, time, json
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")
import torch
from viasd.config import Config
from viasd.models import load_models, lm_logits

cfg = Config(); torch.manual_seed(cfg.seed)
tiers = load_models(cfg)
L = len(tiers.verifier.model.layers)
V = tiers.vocab
gamma = cfg.gamma

dimr = json.load(open(os.path.expanduser("~/viasd/dimr_mask.json")))["keep_mask"]   # len L, True=keep (keeps attn in kept blocks)

def sublayer_skip(n_attn, n_mlp):
    """len-2L skip_set (1=skip): skip attn in the middle n_attn layers, mlp in middle n_mlp."""
    ss = [0] * (2 * L)
    mid = list(range(2, L - 2))
    for i in mid[:n_attn]:
        ss[2 * i] = 1
    for i in mid[:n_mlp]:
        ss[2 * i + 1] = 1
    return ss

# budget-matched-ish to DIMR's 22 skipped blocks: attn-heavy sublayer skip
masks = {
    "full-q":            None,
    "DIMR-block-26":     dimr,                       # keeps attn in 26 kept blocks
    "sub-attn22":        sublayer_skip(22, 0),       # drop attn in 22 layers, keep all mlp
    "sub-attn22-mlp22":  sublayer_skip(22, 22),      # ~match DIMR's total param drop
}
CTX = [256, 1024, 4096, 8192, 16384]

def timeit(fn, reps=6, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0

print(f"L={L} gamma={gamma}  (q' forward over ctx+gamma tokens, ms)", flush=True)
hdr = "ctx     " + "".join(f"{n:>20s}" for n in masks)
print(hdr, flush=True)
results = {}
for ctx in CTX:
    blk = torch.randint(0, V, (1, ctx + gamma), device=tiers.device)
    row = {}
    for nm, mk in masks.items():
        try:
            row[nm] = timeit(lambda: lm_logits(tiers.verifier, blk, mk))
        except RuntimeError as e:
            row[nm] = float("nan"); print(f"  [oom@{ctx}/{nm}] {str(e)[:50]}", flush=True)
    results[ctx] = row
    print(f"{ctx:<8d}" + "".join(f"{row[n]:>19.1f}m" for n in masks), flush=True)
json.dump(results, open(os.path.expanduser("~/bench_logs/latency_sweep.json"), "w"))
# speedup vs DIMR at each ctx (the headline: does sublayer attn-skip pull ahead with context?)
print("\nspeedup of sub-attn22 vs DIMR-block (>1 = sublayer cheaper):", flush=True)
for ctx in CTX:
    r = results[ctx]
    if r["DIMR-block-26"] > 0:
        print(f"  ctx={ctx:<6d} sub-attn22/DIMR = {r['DIMR-block-26']/r['sub-attn22']:.2f}x cheaper", flush=True)
print("[DONE_SWEEP]", flush=True)
