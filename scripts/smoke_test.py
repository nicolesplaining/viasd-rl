"""Fast plumbing check: load models, time tiers, run every decoder on 2 problems
with a short budget. Validates shapes/flow before committing to full runs."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.cost import CostMeter
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.decoding import (FixedThresholdDecider, OracleDecider,
                            check_plain_sd_sequence_equal, greedy_q_generate,
                            plain_sd_generate, via_sd_generate)
from viasd.metrics import is_correct
from viasd.models import load_models, measure_latencies


def main():
    cfg = Config(max_new_tokens=256)
    torch.manual_seed(cfg.seed)
    print("device:", cfg.device, "dtype:", cfg.dtype)
    tiers = load_models(cfg)
    n_layers = len(tiers.verifier.model.layers)
    print(f"verifier layers={n_layers}  keep={sum(tiers.keep_mask)}  mask={tiers.keep_mask}")

    lat = measure_latencies(tiers)
    print(f"latencies (ms): t_p1={lat.t_p1*1e3:.2f} t_qp={lat.t_qp*1e3:.2f} "
          f"t_q={lat.t_q*1e3:.2f} t_q1={lat.t_q1*1e3:.2f}")

    problems = load_gsm8k(2, split="test")
    for q, gold in problems:
        ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
        plen = ids.shape[1]
        print(f"\nQ: {q[:70]}...  gold={gold}  prompt_tokens={plen}")
        check = check_plain_sd_sequence_equal(tiers, ids)
        print(f"  plain_sd == canonical_q: {check.equal}", end="")
        if not check.equal:
            print(f" mismatch_at={check.first_mismatch} ref={check.ref_token} sd={check.test_token}")
        else:
            print(f" tokens={check.ref_len}")
        for name, fn in [
            ("greedy_q", lambda ids, m: greedy_q_generate(tiers, ids, m)),
            ("plain_sd", lambda ids, m: plain_sd_generate(tiers, ids, m)),
            ("via_fixed", lambda ids, m: via_sd_generate(
                tiers, ids, m, FixedThresholdDecider(cfg.theta_accept, cfg.theta_escalate))),
            ("via_oracle", lambda ids, m: via_sd_generate(tiers, ids, m, OracleDecider())),
        ]:
            m = CostMeter()
            out = fn(ids, m)
            text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
            ok = is_correct(text, gold)
            a, r, e = m.tier_fractions()
            print(f"  {name:<11} correct={int(ok)} tokens={m.tokens} rej={m.rejection_rate:.2f} "
                  f"tiers(a/r/e)={a:.2f}/{r:.2f}/{e:.2f} q/tok={m.q_calls_per_token:.2f} "
                  f"speedup={lat.speedup(m):.2f}x")
    print("\nsmoke test OK")


if __name__ == "__main__":
    main()
