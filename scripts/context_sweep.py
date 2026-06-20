"""Validation: does growing context dilute the launch-overhead pollution?

We hold the SD behavior fixed (real plain_sd forward counts from a short GSM8K
run) and re-measure per-forward latency at increasing context lengths. As context
grows, decode work (weight read + KV attention) rises while the ~10ms launch floor
stays fixed, so the MEASURED speedup should climb toward the overhead-corrected /
bandwidth speedup. Prints the overhead fraction of a verifier step at each length.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.cost import CostMeter
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.decoding import plain_sd_generate
from viasd.models import bandwidth_latencies, corrected_latencies, load_models, measure_latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx_lens", type=str, default="256,2048,8192,32768")
    ap.add_argument("--n_meter", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)

    # representative SD forward counts from a real short-context decode
    problems = load_gsm8k(args.n_meter, split="test")
    M = CostMeter()
    for q, _ in problems:
        ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
        m = CostMeter()
        plain_sd_generate(tiers, ids, m)
        M.merge(m)
    print(f"\nplain_sd forward counts (n={args.n_meter}): draft_steps={M.draft_steps} "
          f"q_forwards={M.q_forwards} tokens={M.tokens}\n")

    lat_bw = bandwidth_latencies(tiers)
    print(f"{'ctx_len':>8}{'t_p1':>8}{'t_q1':>8}{'t_q':>8}{'ovh%':>7}{'meas_spd':>10}{'corr_spd':>10}")
    print("-" * 59)
    for ctx in [int(x) for x in args.ctx_lens.split(",")]:
        lat = measure_latencies(tiers, ctx_len=ctx, reps=5, warmup=2)
        latc = corrected_latencies(tiers, lat)
        ovh = 100.0 * (lat.t_q1 - latc.t_q1) / lat.t_q1   # launch floor as % of a verifier step
        print(f"{ctx:>8}{lat.t_p1*1e3:>8.2f}{lat.t_q1*1e3:>8.2f}{lat.t_q*1e3:>8.2f}"
              f"{ovh:>7.1f}{lat.speedup(M):>9.2f}x{latc.speedup(M):>9.2f}x", flush=True)
    print(f"\nbandwidth-model speedup (overhead-free, KV-free ideal): {lat_bw.speedup(M):.2f}x")
    print("expectation: meas_spd rises toward corr_spd as ctx grows (overhead dilutes);\n"
          "the drafter step stays overhead-bound longest, so convergence is partial.")


if __name__ == "__main__":
    main()
