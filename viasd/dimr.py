"""Dynamic Intra-Model Routing (DIMR): offline search for the layer-skip mask z*
that makes the slim-verifier q' best approximate the full verifier q.

Faithful to the paper's Eq. 12 and Eq. 13: pick the mask minimizing the
accumulated KL-style *margin-violation* cost
C(z) = sum_t R^KL_{alpha,beta}(q || q'_z)|_t over a calibration window, and
search the discrete mask space with random exploration plus periodic Bayesian
optimization proposals under a fixed skip ratio.
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


def _candidate_layers(n_layers, keep_first_last):
    return list(range(1, n_layers - 1)) if keep_first_last else list(range(n_layers))


def _mask_key(mask):
    return tuple(bool(x) for x in mask)


def _mask_from_skips(n_layers, skipped):
    keep = [True] * n_layers
    for d in skipped:
        keep[d] = False
    return keep


def _random_mask(n_layers, n_skip, keep_first_last, rng):
    cand = _candidate_layers(n_layers, keep_first_last)
    return _mask_from_skips(n_layers, rng.sample(cand, min(n_skip, len(cand))))


def _one_swap_neighbors(mask, keep_first_last):
    cand = _candidate_layers(len(mask), keep_first_last)
    kept = [i for i in cand if mask[i]]
    skipped = [i for i in cand if not mask[i]]
    for add in skipped:
        for drop in kept:
            m = list(mask)
            m[add], m[drop] = True, False
            yield m


def _mask_features(masks, candidates):
    return torch.tensor([[1.0 if m[i] else 0.0 for i in candidates] for m in masks],
                        dtype=torch.float64)


def _hamming_rbf(x1, x2, length_scale):
    dist = (x1[:, None, :] - x2[None, :, :]).abs().sum(-1)
    return torch.exp(-dist / max(length_scale, 1e-6))


def _expected_improvement(mu, sigma, best):
    sigma = sigma.clamp_min(1e-9)
    improvement = best - mu
    z = improvement / sigma
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=mu.dtype), torch.tensor(1.0, dtype=mu.dtype)
    )
    return improvement * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))


def _bayesopt_mask(observations, n_layers, n_skip, keep_first_last, rng,
                   pool_size=512, top_k=8, length_scale=None, noise=1e-4):
    """Discrete GP-EI proposal over fixed-skip masks.

    The VIA-SD paper does not prescribe a concrete BO package. This implements
    the described operation directly: fit a Gaussian-process surrogate to the
    observed (mask, cost) pairs and evaluate expected improvement on a mixed pool
    of random and local-neighbor candidate masks.
    """
    if len(observations) < 3:
        return _random_mask(n_layers, n_skip, keep_first_last, rng)

    seen = {_mask_key(m) for m, _s in observations}
    pool = {}

    ranked = sorted(observations, key=lambda x: x[1])[:top_k]
    for m, _s in ranked:
        for nb in _one_swap_neighbors(m, keep_first_last):
            key = _mask_key(nb)
            if key not in seen:
                pool[key] = nb
            if len(pool) >= pool_size:
                break
        if len(pool) >= pool_size:
            break

    tries = 0
    while len(pool) < pool_size and tries < pool_size * 20:
        tries += 1
        m = _random_mask(n_layers, n_skip, keep_first_last, rng)
        key = _mask_key(m)
        if key not in seen:
            pool[key] = m
    if not pool:
        return _random_mask(n_layers, n_skip, keep_first_last, rng)

    candidates = _candidate_layers(n_layers, keep_first_last)
    train_masks = [m for m, _s in observations]
    scores = torch.tensor([s for _m, s in observations], dtype=torch.float64)
    y_mean = scores.mean()
    y_std = scores.std(unbiased=False).clamp_min(1e-8)
    y = (scores - y_mean) / y_std

    x_train = _mask_features(train_masks, candidates)
    x_pool = _mask_features(list(pool.values()), candidates)
    length_scale = length_scale or max(1.0, len(candidates) * 0.15)
    k = _hamming_rbf(x_train, x_train, length_scale)
    k = k + torch.eye(k.shape[0], dtype=k.dtype) * noise
    try:
        chol = torch.linalg.cholesky(k)
        alpha_vec = torch.cholesky_solve(y[:, None], chol)[:, 0]
        k_star = _hamming_rbf(x_train, x_pool, length_scale)
        mu = k_star.T @ alpha_vec
        v = torch.cholesky_solve(k_star, chol)
        var = (1.0 - (k_star * v).sum(0)).clamp_min(1e-9)
    except RuntimeError:
        return _random_mask(n_layers, n_skip, keep_first_last, rng)

    best = (scores.min() - y_mean) / y_std
    ei = _expected_improvement(mu, var.sqrt(), best)
    best_idx = int(ei.argmax())
    return list(pool.values())[best_idx]


def search(tiers, calib, skip_ratio, keep_first_last, max_steps=60, bo_period=10,
           patience=20, seed=0, alpha=0.5, beta=0.3, bo_pool=512,
           bo_top_k=8, bo_length_scale=None, bo_noise=1e-4):
    rng = random.Random(seed)
    n_layers = len(tiers.verifier.model.layers)
    n_skip = int(round(skip_ratio * n_layers))

    best = make_keep_mask(n_layers, skip_ratio, keep_first_last)
    best_score = score_mask(tiers, calib, best, alpha, beta)
    observations = [(best, best_score)]
    seen = {_mask_key(best)}
    stale = 0
    print(f"[init evenly-spaced] skip={n_skip}/{n_layers} score={best_score:.5f}", flush=True)

    for step in range(1, max_steps + 1):
        if bo_period > 0 and step % bo_period == 0:
            source = "bayesopt"
            m = _bayesopt_mask(observations, n_layers, n_skip, keep_first_last, rng,
                               pool_size=bo_pool, top_k=bo_top_k,
                               length_scale=bo_length_scale, noise=bo_noise)
        else:
            source = "random"
            m = _random_mask(n_layers, n_skip, keep_first_last, rng)

        retries = 0
        while _mask_key(m) in seen and retries < 100:
            retries += 1
            m = _random_mask(n_layers, n_skip, keep_first_last, rng)
            source = f"{source}+random"
        key = _mask_key(m)
        if key in seen:
            print(f"[step {step:03d} {source}] no unseen masks left in proposal budget", flush=True)
            break
        seen.add(key)

        s = score_mask(tiers, calib, m, alpha, beta)
        observations.append((m, s))
        if s < best_score:
            best, best_score = m, s
            stale = 0
            print(f"[step {step:03d} {source}] new best score={best_score:.5f}", flush=True)
        else:
            stale += 1
            print(f"[step {step:03d} {source}] score={s:.5f} best={best_score:.5f} stale={stale}",
                  flush=True)
        if patience > 0 and stale >= patience:
            print(f"[early stop] best unchanged for {patience} evaluated masks", flush=True)
            break
    return best, best_score, observations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=12)
    ap.add_argument("--skip_ratio", type=float, default=0.45)
    ap.add_argument("--max_steps", type=int, default=60,
                    help="maximum DIMR mask evaluations after the evenly-spaced initializer")
    ap.add_argument("--bo_period", type=int, default=10,
                    help="BayesOpt trigger period theta from the paper's Eq. 13")
    ap.add_argument("--patience", type=int, default=20,
                    help="stop when the best mask is unchanged for this many evaluations")
    ap.add_argument("--bo_pool", type=int, default=512,
                    help="candidate masks scored by the Bayesian acquisition at each BO step")
    ap.add_argument("--bo_top_k", type=int, default=8,
                    help="number of best observed masks whose one-swap neighbors seed the BO pool")
    ap.add_argument("--bo_length_scale", type=float, default=0.0,
                    help="Hamming RBF length scale; 0 uses 15 percent of searchable layers")
    ap.add_argument("--bo_noise", type=float, default=1e-4,
                    help="Gaussian-process diagonal noise")
    ap.add_argument("--n_random", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--hill_steps", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--alpha", type=float, default=0.5)   # paper alpha1 (acceptance)
    ap.add_argument("--beta", type=float, default=0.3)    # paper alpha2 (residual)
    ap.add_argument("--out", type=str, default=str(DEFAULT_LOCAL_RESULTS / "dimr_mask.json"))
    args = ap.parse_args()

    cfg = Config(skip_ratio=args.skip_ratio)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    problems = load_gsm8k(args.n_calib, split="train")
    calib = [build_prompt_ids(tiers.tokenizer, q, tiers.device) for q, _ in problems]

    if args.n_random is not None:
        args.max_steps = args.n_random
    bo_length_scale = args.bo_length_scale if args.bo_length_scale > 0 else None
    mask, score, observations = search(
        tiers, calib, args.skip_ratio, cfg.keep_first_last,
        max_steps=args.max_steps, bo_period=args.bo_period, patience=args.patience,
        seed=cfg.seed, alpha=args.alpha, beta=args.beta, bo_pool=args.bo_pool,
        bo_top_k=args.bo_top_k, bo_length_scale=bo_length_scale, bo_noise=args.bo_noise,
    )
    ensure_parent_dir(args.out)
    with open(args.out, "w") as f:
        json.dump({
            "keep_mask": mask,
            "skip_ratio": args.skip_ratio,
            "score": score,
            "alpha": args.alpha,
            "beta": args.beta,
            "search": "random_search_with_periodic_bayesian_optimization",
            "max_steps": args.max_steps,
            "bo_period": args.bo_period,
            "patience": args.patience,
            "bo_pool": args.bo_pool,
            "bo_top_k": args.bo_top_k,
            "bo_length_scale": bo_length_scale,
            "bo_noise": args.bo_noise,
            "evaluations": len(observations),
        }, f)
    n_keep = sum(mask)
    print(f"saved -> {args.out}  (keep {n_keep}/{len(mask)} layers, score={score:.5f})")


if __name__ == "__main__":
    main()
