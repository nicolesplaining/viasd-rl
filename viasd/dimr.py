"""Dynamic Intra-Model Routing (DIMR): offline search for the layer-skip mask z*
that makes the slim-verifier q' best approximate the full verifier q.

Faithful to the paper's intent (Eq. 12: pick the mask minimizing an accumulated
KL-style cost between q and q'_z over a calibration window). We use mean
KL(q || q'_z) over calibration tokens as the score, and combine random search
with greedy hill-climbing under a fixed skip ratio.
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F

from .config import Config
from .data_gsm8k import build_prompt_ids, load_gsm8k
from .models import load_models, lm_logits, make_keep_mask


@torch.no_grad()
def score_mask(tiers, calib, mask):
    """Mean KL(q || q'_mask) over calibration tokens (lower is better)."""
    total, ntok = 0.0, 0
    for ids in calib:
        q = lm_logits(tiers.verifier, ids, None)[0]
        qp = lm_logits(tiers.verifier, ids, mask)[0]
        lq = F.log_softmax(q, dim=-1)
        lqp = F.log_softmax(qp, dim=-1)
        kl = (lq.exp() * (lq - lqp)).sum(-1)  # per position
        total += float(kl.sum()); ntok += kl.numel()
    return total / max(ntok, 1)


def _random_mask(n_layers, n_skip, keep_first_last):
    keep = [True] * n_layers
    cand = list(range(1, n_layers - 1)) if keep_first_last else list(range(n_layers))
    for d in random.sample(cand, min(n_skip, len(cand))):
        keep[d] = False
    return keep


def search(tiers, calib, skip_ratio, keep_first_last, n_random=30, hill_steps=40, seed=0):
    random.seed(seed)
    n_layers = len(tiers.verifier.model.layers)
    n_skip = int(round(skip_ratio * n_layers))

    best = make_keep_mask(n_layers, skip_ratio, keep_first_last)
    best_score = score_mask(tiers, calib, best)
    print(f"[init evenly-spaced] skip={n_skip}/{n_layers} score={best_score:.5f}", flush=True)

    for i in range(n_random):
        m = _random_mask(n_layers, n_skip, keep_first_last)
        s = score_mask(tiers, calib, m)
        if s < best_score:
            best, best_score = m, s
            print(f"[random {i+1}] new best score={best_score:.5f}", flush=True)

    # greedy hill-climb: swap one kept<->skipped layer if it improves
    cand = list(range(1, n_layers - 1)) if keep_first_last else list(range(n_layers))
    for step in range(hill_steps):
        improved = False
        kept = [i for i in cand if best[i]]
        skipped = [i for i in cand if not best[i]]
        random.shuffle(kept); random.shuffle(skipped)
        for a in skipped:
            for b in kept:
                m = list(best); m[a], m[b] = True, False
                s = score_mask(tiers, calib, m)
                if s < best_score:
                    best, best_score = m, s
                    improved = True
                    print(f"[hill {step+1}] swap skip{b}->keep{a} score={best_score:.5f}", flush=True)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=12)
    ap.add_argument("--skip_ratio", type=float, default=0.45)
    ap.add_argument("--n_random", type=int, default=30)
    ap.add_argument("--hill_steps", type=int, default=40)
    ap.add_argument("--out", type=str, default="dimr_mask.json")
    args = ap.parse_args()

    cfg = Config(skip_ratio=args.skip_ratio)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    problems = load_gsm8k(args.n_calib, split="train")
    calib = [build_prompt_ids(tiers.tokenizer, q, tiers.device) for q, _ in problems]

    mask, score = search(tiers, calib, args.skip_ratio, cfg.keep_first_last,
                         n_random=args.n_random, hill_steps=args.hill_steps, seed=cfg.seed)
    with open(args.out, "w") as f:
        json.dump({"keep_mask": mask, "skip_ratio": args.skip_ratio, "score": score}, f)
    n_keep = sum(mask)
    print(f"saved -> {args.out}  (keep {n_keep}/{len(mask)} layers, score={score:.5f})")


if __name__ == "__main__":
    main()
