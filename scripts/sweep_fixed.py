"""Tune via_fixed's thresholds fairly: sweep (theta_accept, theta_escalate) and
report accuracy + q-calls/token per config, so we compare via_rl against the BEST
fixed-threshold VIA-SD, not an arbitrary one. Reports greedy_q once as reference.

Only accuracy and q-calls/token are reported (both hardware-independent), so no
latency measurement is needed -> faster."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.cost import CostMeter
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.decoding import FixedThresholdDecider, greedy_q_generate, via_sd_generate
from viasd.metrics import is_correct
from viasd.models import load_models


def run(tiers, problems, gen_fn):
    agg, correct = CostMeter(), 0
    for q, gold in problems:
        ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
        plen = ids.shape[1]
        m = CostMeter()
        out = gen_fn(ids, m)
        correct += int(is_correct(tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True), gold))
        agg.merge(m)
    return agg, correct / len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--accepts", type=str, default="0.4,0.6,0.8,0.9,0.95")
    ap.add_argument("--escalates", type=str, default="0.2,0.4")
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    problems = load_gsm8k(args.n_eval, split="test")

    _, acc_g = run(tiers, problems, lambda ids, m: greedy_q_generate(tiers, ids, m))
    print(f"\nreference greedy_q accuracy = {acc_g:.3f}  (n={args.n_eval})\n")

    accepts = [float(x) for x in args.accepts.split(",")]
    escalates = [float(x) for x in args.escalates.split(",")]
    print(f"{'theta_acc':>10}{'theta_esc':>10}{'acc':>8}{'q/tok':>8}{'accept':>8}{'escal':>8}{'rej':>8}")
    print("-" * 60)
    for ta in accepts:
        for te in escalates:
            if te >= ta:
                continue
            agg, acc = run(tiers, problems,
                           lambda ids, m, ta=ta, te=te: via_sd_generate(
                               tiers, ids, m, FixedThresholdDecider(ta, te)))
            a, r, e = agg.tier_fractions()
            print(f"{ta:>10.2f}{te:>10.2f}{acc:>8.3f}{agg.q_calls_per_token:>8.3f}"
                  f"{a:>8.3f}{e:>8.3f}{agg.rejection_rate:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
