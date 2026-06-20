#!/usr/bin/env python3
# gen_knapspec_qprime.py — produce a "KnapSpec-decided q'" keep_mask by running Nicole's / the
# official KnapSpec generator (KnapspecGenerator) on GSM8K calibration prompts and extracting the
# DP-selected skip_set. The DP (Knapspec.optimize) chooses which attn/MLP sublayers to skip to
# maximize tokens-per-time under a budget. We aggregate its choices over several prompts, then
# collapse the 2L sublayer skip_set into a whole-block keep_mask (the format her q' uses).
# Emits TWO masks: (1) KnapSpec's NATURAL operating point (its own chosen #skips), and
# (2) budget-MATCHED to DIMR (keep exactly 26/48) for an apples-to-apples selector comparison.
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
DIMR_KEEP = 26          # match DIMR's 26/48 for the budget-matched mask
N_CALIB = 8             # calibration prompts
MAX_NEW = 192
OUT_DIR = os.path.expanduser("~/viasd")

print(f"[load] {MID}", flush=True)
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to("cuda").eval()
L = len(model.model.layers)
env = Env(model=model, tok=tok, device="cuda", eos_id=tok.eos_token_id, pad_id=tok.pad_token_id)
print("[profile] solving C1,C2,C3 ...", flush=True)
c1, c2, c3 = profile_model(env.model)
coeff = (c1, c2, c3)
print(f"[profile] C1={c1} C2={c2} C3={c3}", flush=True)

# generous sublayer budget so the DP can choose freely; it stops at its TPT optimum.
gen = KnapspecGenerator(env, gamma=4, skip_budget_M=2 * L, optimize_interval=64,
                        coefficients=coeff, sim_threshold=0.5)
ds = load_dataset("openai/gsm8k", "main", split="test")

freq = [0.0] * (2 * L)      # per-sublayer skip frequency
tot_skip = 0.0; ncalls = 0
for i in range(N_CALIB):
    msgs = [{"role": "user", "content": ds[i]["question"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen.generate(prompt, max_new_tokens=MAX_NEW)
    ss = gen.knapspec_model.skip_set      # 2L list, 1=skip
    for j, v in enumerate(ss):
        freq[j] += v
    tot_skip += sum(ss); ncalls += 1
    print(f"[calib {i}] knapspec skip_count={sum(ss)}/{2*L}", flush=True)

# per-block skip score (sum of attn+mlp skip freq); higher => more skippable
score = [freq[2 * i] + freq[2 * i + 1] for i in range(L)]
order = sorted(range(L), key=lambda i: -score[i])    # most-skippable first

# (1) NATURAL: skip KnapSpec's own chosen avg number of blocks (= avg sublayer skips / 2)
nat_skip_blocks = max(1, round((tot_skip / ncalls) / 2))
keep_nat = [True] * L
for i in order[:nat_skip_blocks]:
    keep_nat[i] = False

# (2) MATCHED: keep exactly DIMR_KEEP blocks (skip the L-DIMR_KEEP most-skippable)
keep_match = [True] * L
for i in order[:(L - DIMR_KEEP)]:
    keep_match[i] = False

for name, km in [("knapspec_qprime_natural", keep_nat), ("knapspec_qprime_match26", keep_match)]:
    p = os.path.join(OUT_DIR, name + ".json")
    json.dump({"keep_mask": [bool(x) for x in km]}, open(p, "w"))
    print(f"[WROTE] {p}  keep={sum(km)}/{L}", flush=True)
print("[DONE]", flush=True)
