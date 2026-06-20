"""RL refinement of the gating policy (REINFORCE with a moving-average baseline).

Reward is a constrained-style scalarization: maximize task correctness while
penalizing estimated latency. The policy's actions change which tokens are
emitted, so the quality payoff (GSM8K correctness) is delayed and sequence-level
-- exactly the regime where RL beats per-token supervision.

    reward = R_correct * correct - lambda * (estimated_latency_per_token / t_q1)

The cost term is in units of a full-verifier autoregressive step, so the greedy-q
baseline has cost 1.0/token and any speedup shows up as cost < 1.0.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from .config import Config
from .cost import CostMeter
from .data_gsm8k import build_prompt_ids, load_gsm8k
from .decoding import PolicyDecider, via_sd_generate
from .metrics import is_correct
from .models import load_models, measure_latencies
from .policy import GatingPolicy, load_policy, save_policy


def rollout(tiers, policy, q, gold, lat, lam, r_correct):
    """Dense reward: per-token agreement with the full verifier's greedy choice
    (measured at training time via record_qmatch, not charged to cost), plus a
    small terminal-correctness bonus, minus a latency penalty.

        reward = match_rate + r_correct * correct - lam * (latency_per_tok / t_q1)
    """
    ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
    plen = ids.shape[1]
    samples = []
    m = CostMeter()
    decider = PolicyDecider(policy, sample=True)
    out = via_sd_generate(tiers, ids, m, decider, collect=samples, record_qmatch=True)
    text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
    correct = float(is_correct(text, gold))
    matches = [s.match for s in samples if s.match >= 0]
    match_rate = sum(matches) / len(matches) if matches else 0.0
    cost_per_tok = (lat.estimate(m) / max(m.tokens, 1)) / lat.t_q1
    reward = match_rate + r_correct * correct - lam * cost_per_tok
    return samples, reward, correct, match_rate, cost_per_tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="policy_imitation.pt")
    ap.add_argument("--out", type=str, default="policy_rl.pt")
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)        # trajectories per update
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.3)      # latency penalty weight
    ap.add_argument("--r_correct", type=float, default=0.5)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="rl_ckpt.pt")   # resumable checkpoint
    ap.add_argument("--resume", type=str, default="")           # path to resume from
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=1)         # constant per-iter logging
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    lat = measure_latencies(tiers)
    print(f"latencies (ms): t_p1={lat.t_p1*1e3:.2f} t_qp={lat.t_qp*1e3:.2f} "
          f"t_q={lat.t_q*1e3:.2f} t_q1={lat.t_q1*1e3:.2f}", flush=True)

    device = cfg.device
    try:
        policy = load_policy(args.init, device)
        print(f"warm-started from {args.init}")
    except (FileNotFoundError, RuntimeError):
        policy = GatingPolicy().to(device)
        print("cold-started policy")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    problems = load_gsm8k(args.n_train, split="train")
    baseline = 0.0
    pi = 0
    start_iter = 0
    resume_path = args.resume or (args.ckpt if os.path.exists(args.ckpt) else "")
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["opt"])
        baseline = ck["baseline"]; pi = ck["pi"]; start_iter = ck["iter"]
        print(f"RESUMED from {resume_path} at iter {start_iter} (baseline={baseline:.3f})", flush=True)

    def save_ckpt(it_done):
        torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                    "baseline": baseline, "pi": pi, "iter": it_done,
                    "args": vars(args)}, args.ckpt)
        save_policy(policy, args.out)

    for it in range(start_iter, args.iters):
        batch_loss = 0.0
        stats = {"reward": 0.0, "correct": 0.0, "match": 0.0, "cost": 0.0, "rej": 0.0}
        for _ in range(args.batch):
            q, gold = problems[pi % len(problems)]; pi += 1
            samples, reward, correct, match_rate, cost, m = rollout(
                tiers, policy, q, gold, lat, args.lam, args.r_correct)
            baseline = 0.95 * baseline + 0.05 * reward
            adv = reward - baseline
            if not samples:
                continue
            feats = torch.stack([s.feats for s in samples]).to(device)
            acts = torch.tensor([s.action for s in samples], device=device)
            logits = policy(feats)
            logp = F.log_softmax(logits, dim=-1)
            chosen = logp[torch.arange(len(acts)), acts]
            probs = logp.exp()
            entropy = -(probs * logp).sum(-1).mean()
            loss = -(adv * chosen.sum()) / len(acts) - args.entropy * entropy
            batch_loss = batch_loss + loss
            stats["reward"] += reward; stats["correct"] += correct
            stats["match"] += match_rate; stats["cost"] += cost; stats["rej"] += m.rejection_rate
        opt.zero_grad(); (batch_loss / args.batch).backward(); opt.step()
        b = args.batch
        if (it + 1) % args.log_every == 0:
            print(f"iter {it+1}/{args.iters}: reward={stats['reward']/b:.3f} "
                  f"match={stats['match']/b:.3f} correct={stats['correct']/b:.3f} "
                  f"cost/tok={stats['cost']/b:.3f} rej={stats['rej']/b:.3f} "
                  f"baseline={baseline:.3f}", flush=True)
        if (it + 1) % args.ckpt_every == 0:
            save_ckpt(it + 1)
            print(f"  [ckpt @ iter {it+1} -> {args.ckpt}]", flush=True)
    save_ckpt(args.iters)
    print(f"saved RL policy -> {args.out} ; checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
