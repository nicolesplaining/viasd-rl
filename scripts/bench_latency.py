"""Measure per-forward latency eager vs torch.compile, to quantify the overhead fix.

The wall-clock problem is a per-call overhead floor (t_q for 261 tokens ~= t_q1 for
1 token => GPU idle, launch-bound). torch.compile (esp. reduce-overhead = CUDA graphs)
should cut that. This isolates the effect before committing training to it."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.models import compile_models, load_models, measure_latencies


def row(tag, lat):
    print(f"{tag:<20} t_p1={lat.t_p1*1e3:7.2f}  t_qp={lat.t_qp*1e3:7.2f}  "
          f"t_q={lat.t_q*1e3:7.2f}  t_q1={lat.t_q1*1e3:7.2f}  (ms)")
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="reduce-overhead",
                    help="torch.compile mode: default | reduce-overhead | max-autotune")
    args = ap.parse_args()

    cfg = Config()
    tiers = load_models(cfg)
    print(f"models: {cfg.drafter_name} -> {cfg.verifier_name}\n")

    eager = row("EAGER", measure_latencies(tiers, warmup=3))

    print(f"\ncompiling (mode={args.mode}) ... first measurement includes compile warmup")
    compile_models(tiers, mode=args.mode)
    comp = row("COMPILED", measure_latencies(tiers, warmup=10))

    print("\nspeedup (eager / compiled):")
    print(f"  drafter 1-tok : {eager.t_p1/comp.t_p1:5.2f}x")
    print(f"  q' block      : {eager.t_qp/comp.t_qp:5.2f}x")
    print(f"  q  block      : {eager.t_q/comp.t_q:5.2f}x")
    print(f"  q  1-tok      : {eager.t_q1/comp.t_q1:5.2f}x")
    print("\n(if drafter 1-tok and q 1-tok drop a lot, the overhead floor is fixed "
          "and speculative decoding will show real wall-clock speedup.)")


if __name__ == "__main__":
    main()
