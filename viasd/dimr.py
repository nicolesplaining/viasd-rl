"""Dynamic Intra-Model Routing (DIMR): offline search for the layer-skip mask z*
that makes the slim-verifier q' best approximate the full verifier q.

Faithful to the paper's Eq. 12: pick the mask minimizing the accumulated KL-style
*margin-violation* cost C(z) = sum_t R^KL_{alpha,beta}(q || q'_z)|_t over a
calibration window. R^KL is NOT raw KL(q||q'); it is the token-wise log-margin
violation cost of Eq. 11, which is what actually governs the rejection rate in
the q'->q verification gate. We combine random search with greedy hill-climbing
under a fixed skip ratio.
"""
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F

from .config import Config
from .data_gsm8k import build_prompt_ids, load_gsm8k
from .models import load_models, lm_logits, make_keep_mask
from .paths import DEFAULT_LOCAL_RESULTS, ensure_parent_dir


@torch.no_grad()
def score_mask(tiers, calib, mask, alpha=0.5, beta=0.3):
    """Mean per-token R^KL_{alpha,beta}(q || q'_mask) over calibration tokens
    (lower is better) -- the paper's Eq. 11 margin-violation cost for the q'->q
    gate, where the slim-verifier q' plays the drafter role and the full q verifies.

    Eq. 11 (with p := q', the stage's drafter):
        z1(v) = log(1-alpha) + log q'(v) - log q(v)   # acceptance-threshold violation
        z2(v) = log(beta)    + log q'(v) - log q(v)    # residual-replacement violation
        cost  = sum_v q'(v) ReLU(z1(v)) + sum_v q(v) ReLU(z2(v))
    (alpha, beta) default to the paper's reported (alpha1=0.5, alpha2=0.3).
    """
    log_1ma = math.log(1.0 - alpha)
    log_beta = math.log(beta)
    total, ntok = 0.0, 0
    for ids in calib:
        q = lm_logits(tiers.verifier, ids, None)[0]
        qp = lm_logits(tiers.verifier, ids, mask)[0]
        lq = F.log_softmax(q, dim=-1)       # log q
        lqp = F.log_softmax(qp, dim=-1)     # log q'
        margin = lqp - lq                   # log(q'/q) per vocab entry
        z1 = (log_1ma + margin).clamp(min=0.0)   # ReLU(z1)
        z2 = (log_beta + margin).clamp(min=0.0)  # ReLU(z2)
        cost = (lqp.exp() * z1).sum(-1) + (lq.exp() * z2).sum(-1)  # per position
        total += float(cost.sum()); ntok += cost.numel()
    return total / max(ntok, 1)


def _random_mask(n_layers, n_skip, keep_first_last):
    keep = [True] * n_layers
    cand = list(range(1, n_layers - 1)) if keep_first_last else list(range(n_layers))
    for d in random.sample(cand, min(n_skip, len(cand))):
        keep[d] = False
    return keep


def search(tiers, calib, skip_ratio, keep_first_last, n_random=30, hill_steps=40,
           seed=0, alpha=0.5, beta=0.3):
    random.seed(seed)
    n_layers = len(tiers.verifier.model.layers)
    n_skip = int(round(skip_ratio * n_layers))

    best = make_keep_mask(n_layers, skip_ratio, keep_first_last)
    best_score = score_mask(tiers, calib, best, alpha, beta)
    print(f"[init evenly-spaced] skip={n_skip}/{n_layers} score={best_score:.5f}", flush=True)

    for i in range(n_random):
        m = _random_mask(n_layers, n_skip, keep_first_last)
        s = score_mask(tiers, calib, m, alpha, beta)
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
                s = score_mask(tiers, calib, m, alpha, beta)
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
    ap.add_argument("--alpha", type=float, default=0.5)   # paper alpha1 (acceptance)
    ap.add_argument("--beta", type=float, default=0.3)    # paper alpha2 (residual)
    ap.add_argument("--out", type=str, default=str(DEFAULT_LOCAL_RESULTS / "dimr_mask.json"))
    args = ap.parse_args()

    cfg = Config(skip_ratio=args.skip_ratio)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    problems = load_gsm8k(args.n_calib, split="train")
    calib = [build_prompt_ids(tiers.tokenizer, q, tiers.device) for q, _ in problems]

    mask, score = search(tiers, calib, args.skip_ratio, cfg.keep_first_last,
                         n_random=args.n_random, hill_steps=args.hill_steps, seed=cfg.seed,
                         alpha=args.alpha, beta=args.beta)
    ensure_parent_dir(args.out)
    with open(args.out, "w") as f:
        json.dump({"keep_mask": mask, "skip_ratio": args.skip_ratio, "score": score,
                   "alpha": args.alpha, "beta": args.beta}, f)
    n_keep = sum(mask)
    print(f"saved -> {args.out}  (keep {n_keep}/{len(mask)} layers, score={score:.5f})")


if __name__ == "__main__":
    main()
