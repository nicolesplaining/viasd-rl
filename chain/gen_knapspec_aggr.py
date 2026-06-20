#!/usr/bin/env python3
# gen_knapspec_masks.py — "KnapSpec-DP decides q'" at multiple keep-budgets. Runs the official
# KnapspecGenerator DP on GSM8K calibration (reuses ~/work/knapspec), aggregates its per-sublayer
# skip choices into a skippability ranking, then emits a whole-block keep_mask at each budget by
# skipping the (L-keep) most-skippable blocks. One calibration -> all budgets (consistent ranking).
import os, sys, json
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import torch
KS = os.path.expanduser("~/work/knapspec"); sys.path.insert(0, KS)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import Env
from profile_modules import profile_model
from self_speculation_strategy.knapspec_generator import KnapspecGenerator
from datasets import load_dataset

MID = os.environ.get("VIASD_VERIFIER", "Qwen/Qwen2.5-14B-Instruct")
OUT = os.path.expanduser("~/viasd")
L = 48
BUDGETS = [8, 10, 12, 14]
N_CALIB = 8
MAX_NEW = 192

print(f"[load] {MID}", flush=True)
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to("cuda").eval()
assert len(model.model.layers) == L, f"expected {L} layers, got {len(model.model.layers)}"
env = Env(model=model, tok=tok, device="cuda", eos_id=tok.eos_token_id, pad_id=tok.pad_token_id)
print("[profile] C1,C2,C3 ...", flush=True)
c1, c2, c3 = profile_model(env.model)
gen = KnapspecGenerator(env, gamma=4, skip_budget_M=2 * L, optimize_interval=64,
                        coefficients=(c1, c2, c3), sim_threshold=0.5)
ds = load_dataset("openai/gsm8k", "main", split="test")

freq = [0.0] * (2 * L)
for i in range(N_CALIB):
    msgs = [{"role": "user", "content": ds[i]["question"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen.generate(prompt, max_new_tokens=MAX_NEW)
    ss = gen.knapspec_model.skip_set
    for j, v in enumerate(ss):
        freq[j] += v
    print(f"[calib {i}] knapspec skip_count={sum(ss)}/{2*L}", flush=True)

# per-block skippability score (attn+mlp skip freq); higher => KnapSpec prefers to skip it
score = [freq[2 * i] + freq[2 * i + 1] for i in range(L)]
order = sorted(range(L), key=lambda i: -score[i])   # most-skippable first
for keep in BUDGETS:
    km = [True] * L
    for i in order[:(L - keep)]:
        km[i] = False
    json.dump({"keep_mask": [bool(x) for x in km]}, open(f"{OUT}/knapspec_keep{keep}.json", "w"))
    print(f"[keep{keep}] KnapSpec-DP q' (keep {sum(km)}/{L})", flush=True)
print("[DONE_KNAP]", flush=True)
