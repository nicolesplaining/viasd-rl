"""Benchmark: greedy-q vs plain SD vs fixed-threshold VIA-SD vs learned-policy VIA-SD.

Reports task accuracy, rejection rate, tier distribution, full-verifier calls per
token, and estimated speedup (from per-forward latencies measured on this GPU).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.cost import CostMeter
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.decoding import (FixedThresholdDecider, PolicyDecider,
                            greedy_q_generate, plain_sd_generate, via_sd_generate)
from viasd.metrics import is_correct
from viasd.models import load_models, measure_latencies
from viasd.policy import load_policy


def run_method(tiers, problems, gen_fn):
    agg = CostMeter()
    correct = 0
    for q, gold in problems:
        ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
        plen = ids.shape[1]
        m = CostMeter()
        out = gen_fn(ids, m)
        text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        correct += int(is_correct(text, gold))
        agg.merge(m)
    return agg, correct / len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", type=str, default="")
    ap.add_argument("--policy_imitation", type=str, default="policy_imitation.pt")
    ap.add_argument("--policy_rl", type=str, default="policy_rl.pt")
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    print("keep_mask:", tiers.keep_mask, f"(keep {sum(tiers.keep_mask)}/{len(tiers.keep_mask)})")
    lat = measure_latencies(tiers)
    print(f"latencies (ms): t_p1={lat.t_p1*1e3:.2f} t_qp={lat.t_qp*1e3:.2f} "
          f"t_q={lat.t_q*1e3:.2f} t_q1={lat.t_q1*1e3:.2f}\n", flush=True)

    problems = load_gsm8k(args.n_eval, split=args.split)

    methods = {
        "greedy_q": lambda ids, m: greedy_q_generate(tiers, ids, m),
        "plain_sd": lambda ids, m: plain_sd_generate(tiers, ids, m),
        "via_fixed": lambda ids, m: via_sd_generate(
            tiers, ids, m, FixedThresholdDecider(cfg.theta_accept, cfg.theta_escalate)),
    }
    if os.path.exists(args.policy_imitation):
        pol = load_policy(args.policy_imitation, cfg.device)
        methods["via_imit"] = lambda ids, m: via_sd_generate(tiers, ids, m, PolicyDecider(pol))
    if os.path.exists(args.policy_rl):
        polr = load_policy(args.policy_rl, cfg.device)
        methods["via_rl"] = lambda ids, m: via_sd_generate(tiers, ids, m, PolicyDecider(polr))

    rows = []
    for name, fn in methods.items():
        print(f"running {name} ...", flush=True)
        agg, acc = run_method(tiers, problems, fn)
        a, r, e = agg.tier_fractions()
        rows.append((name, acc, agg.rejection_rate, a, r, e,
                     agg.q_calls_per_token, lat.speedup(agg)))

    hdr = f"{'method':<11}{'acc':>7}{'rej':>8}{'accept':>8}{'regen':>8}{'escal':>8}{'q/tok':>8}{'speedup':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, acc, rej, a, r, e, qpt, sp in rows:
        print(f"{name:<11}{acc:>7.3f}{rej:>8.3f}{a:>8.3f}{r:>8.3f}{e:>8.3f}{qpt:>8.3f}{sp:>8.2f}x")


if __name__ == "__main__":
    main()
