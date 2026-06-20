#!/usr/bin/env python3
# gen_knapspec_sublayer.py — KnapSpec-DP q' at NATIVE sublayer (attn/MLP) granularity.
# Runs the official KnapspecGenerator DP on GSM8K calibration, aggregates its per-sublayer skip
# choices, and emits len-2L skip_sets (1=skip) at a few cost points (skip the K most-skippable
# sublayers). This KEEPS the attn/MLP decoupling DIMR can't express -> the real Leg-3 lever.
import os, sys, json
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import torch
KS = os.path.expanduser("~/work/knapspec"); sys.path.insert(0, KS)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import Env
from profile_modules import profile_model
from self_speculation_strategy.knapspec_generator import KnapspecGenerator
from datasets import load_dataset

MID = os.environ.get("VIASD_VERIFIER", "Qwen/Qwen2.5-14B-Instruct")
OUT = os.path.expanduser("~/viasd"); L = 48; N_CALIB = 8; MAX_NEW = 192
SKIP_BUDGETS = [30, 44, 56, 70]   # # of sublayers skipped (of 96); 44 ~= DIMR's 22 blocks

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
assert len(model.model.layers) == L
env = Env(model=model, tok=tok, device="cuda", eos_id=tok.eos_token_id, pad_id=tok.pad_token_id)
c1, c2, c3 = profile_model(env.model)
gen = KnapspecGenerator(env, gamma=4, skip_budget_M=2 * L, optimize_interval=64, coefficients=(c1, c2, c3), sim_threshold=0.5)
ds = load_dataset("openai/gsm8k", "main", split="test")

freq = [0.0] * (2 * L)
for i in range(N_CALIB):
    msgs = [{"role": "user", "content": ds[i]["question"]}]
    gen.generate(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), max_new_tokens=MAX_NEW)
    for j, v in enumerate(gen.knapspec_model.skip_set):
        freq[j] += v
    print(f"[calib {i}] skip={sum(gen.knapspec_model.skip_set)}/{2*L} (attn {sum(gen.knapspec_model.skip_set[0::2])}/mlp {sum(gen.knapspec_model.skip_set[1::2])})", flush=True)

order = sorted(range(2 * L), key=lambda j: -freq[j])   # most-skippable sublayer first
for K in SKIP_BUDGETS:
    ss = [0] * (2 * L)
    for j in order[:K]:
        ss[j] = 1
    n_attn = sum(ss[0::2]); n_mlp = sum(ss[1::2])
    json.dump({"keep_mask": ss}, open(f"{OUT}/knapspec_sub{K}.json", "w"))
    print(f"[sub{K}] skip {K}/{2*L} sublayers (attn {n_attn}/mlp {n_mlp})", flush=True)
print("[DONE_SUB]", flush=True)
