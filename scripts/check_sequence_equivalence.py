#!/usr/bin/env python3
"""Assert that plain speculative decoding matches canonical greedy-q exactly."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viasd.config import Config
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
from viasd.decoding import check_plain_sd_sequence_equal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", default="")
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)

    from viasd.models import load_models

    tiers = load_models(cfg)
    problems = load_gsm8k(args.n_eval, split=args.split)
    for i, (question, _gold) in enumerate(problems):
        ids = build_prompt_ids(tiers.tokenizer, question, tiers.device)
        check = check_plain_sd_sequence_equal(tiers, ids)
        if not check.equal:
            print(
                f"FAILED problem={i} mismatch_at={check.first_mismatch} "
                f"ref_tok={check.ref_token} plain_sd_tok={check.test_token} "
                f"ref_len={check.ref_len} plain_sd_len={check.test_len}",
                flush=True,
            )
            raise SystemExit(1)
        print(f"ok problem={i} generated_tokens={check.ref_len}", flush=True)
    print(f"plain_sd sequence equality OK on {len(problems)} prompts")


if __name__ == "__main__":
    main()
