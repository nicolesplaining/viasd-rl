#!/usr/bin/env python3
# gen_dimr_masks.py — DIMR (KL-margin) + evenly-spaced q' masks at multiple keep-budgets,
# reusing Nicole's viasd.dimr.search and make_keep_mask (run from ~/viasd). One model load,
# one calibration window, all budgets -> directly comparable to the KnapSpec-DP masks.
import os, sys, json
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import torch
from viasd.config import Config
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.models import load_models, make_keep_mask
from viasd.dimr import search

OUT = os.path.expanduser("~/viasd")
L = 48
BUDGETS = [20, 26, 32, 38]   # q' layers kept (of 48)
N_CALIB = 12

cfg = Config(skip_ratio=0.45)
torch.manual_seed(cfg.seed)
print("[load] viasd tiers (drafter + 14B verifier)", flush=True)
tiers = load_models(cfg)
problems = load_gsm8k(N_CALIB, split="train")
calib = [build_prompt_ids(tiers.tokenizer, q, tiers.device) for q, _ in problems]
print(f"[calib] {len(calib)} GSM8K-train prompts", flush=True)

for keep in BUDGETS:
    sr = (L - keep) / L
    ev = make_keep_mask(L, sr, cfg.keep_first_last)
    json.dump({"keep_mask": [bool(x) for x in ev]}, open(f"{OUT}/evenly_keep{keep}.json", "w"))
    mask, score = search(tiers, calib, sr, cfg.keep_first_last)
    json.dump({"keep_mask": [bool(x) for x in mask], "score": score},
              open(f"{OUT}/dimr_keep{keep}.json", "w"))
    print(f"[keep{keep}] evenly(keep {sum(ev)}) + DIMR(keep {sum(mask)}, KLcost={score:.4f})", flush=True)
print("[DONE_DIMR]", flush=True)
