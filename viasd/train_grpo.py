"""GRPO trainer for the per-token gating policy.

Vs. train_rl.py (REINFORCE + global EMA baseline), GRPO:
  - samples a GROUP of G rollouts per prompt and uses the group-normalized reward
    A_i = (r_i - mean_group) / (std_group + eps) as the advantage -- this cancels
    per-problem difficulty variance (the big noise source with batch-4 REINFORCE),
  - applies a PPO-clipped surrogate over several inner epochs,
  - regularizes with KL to a frozen reference (the imitation policy),
  - needs no value network (advantage is group-relative).

Reward per rollout (dense): match_rate + r_correct*correct - lam*(latency/t_q1).

Logging: per-iter console line + JSONL with reward mean/std, match, correct,
cost/tok, rejection, advantage std, KL, entropy, clip fraction, grad norm, sec/iter.
Checkpoints (policy+optimizer+iter+problem-index) every ckpt_every iters; auto-resumes
from --ckpt. Optional torch.compile to attack the per-call overhead floor.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .config import Config
from .cost import CostMeter
from .data_gsm8k import build_prompt_ids, load_gsm8k
from .decoding import PolicyDecider, via_sd_generate
from .metrics import is_correct
from .models import compile_models, load_models, measure_latencies
from .paths import DEFAULT_LOCAL_RESULTS, ensure_parent_dir
from .policy import load_policy, save_policy


def rollout(tiers, policy, q, gold, lat, lam, r_correct, match_weight=1.0):
    ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
    plen = ids.shape[1]
    samples = []
    m = CostMeter()
    out = via_sd_generate(tiers, ids, m, PolicyDecider(policy, sample=True),
                          collect=samples, record_qmatch=True)
    text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
    correct = float(is_correct(text, gold))
    matches = [s.match for s in samples if s.match >= 0]
    match_rate = sum(matches) / len(matches) if matches else 0.0
    cost_per_tok = (lat.estimate(m) / max(m.tokens, 1)) / lat.t_q1
    reward = match_weight * match_rate + r_correct * correct - lam * cost_per_tok
    return samples, reward, correct, match_rate, cost_per_tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(DEFAULT_LOCAL_RESULTS / "policy_imitation.pt"))
    ap.add_argument("--out", default=str(DEFAULT_LOCAL_RESULTS / "policy_grpo.pt"))
    ap.add_argument("--ckpt", default=str(DEFAULT_LOCAL_RESULTS / "grpo_ckpt.pt"))
    ap.add_argument("--resume", default="")
    ap.add_argument("--jsonl", default=str(DEFAULT_LOCAL_RESULTS / "grpo_log.jsonl"))
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--group_size", type=int, default=6)        # G rollouts per prompt
    ap.add_argument("--prompts_per_iter", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)            # PPO inner epochs
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--r_correct", type=float, default=0.5)
    ap.add_argument("--match_weight", type=float, default=1.0)  # set ~0 for correctness-focused RL
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", default="")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default="default")
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)
    device = cfg.device
    tiers = load_models(cfg)
    print("keep_mask:", sum(tiers.keep_mask), "/", len(tiers.keep_mask), flush=True)
    if args.compile:
        print(f"torch.compile (mode={args.compile_mode}) ...", flush=True)
        t0 = time.time(); compile_models(tiers, args.compile_mode)
        print(f"  (compile wrappers attached in {time.time()-t0:.1f}s; graphs build on first calls)", flush=True)
    lat = measure_latencies(tiers, warmup=10 if args.compile else 3)
    print(f"latencies(ms): t_p1={lat.t_p1*1e3:.2f} t_qp={lat.t_qp*1e3:.2f} "
          f"t_q={lat.t_q*1e3:.2f} t_q1={lat.t_q1*1e3:.2f}", flush=True)

    policy = load_policy(args.init, device)             # warm start
    ref = load_policy(args.init, device)                # frozen KL reference
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    problems = load_gsm8k(args.n_train, split="train")
    pi, start_iter = 0, 0
    resume_path = args.resume or (args.ckpt if os.path.exists(args.ckpt) else "")
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["opt"])
        pi, start_iter = ck["pi"], ck["iter"]
        print(f"RESUMED from {resume_path} at iter {start_iter}", flush=True)

    def save_ckpt(it_done):
        ensure_parent_dir(args.ckpt)
        torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                    "pi": pi, "iter": it_done, "args": vars(args)}, args.ckpt)
        save_policy(policy, args.out)

    ensure_parent_dir(args.jsonl)
    jl = open(args.jsonl, "a")

    for it in range(start_iter, args.iters):
        t_iter = time.time()
        feats_buf, act_buf, adv_buf = [], [], []
        r_all, match_all, correct_all, cost_all, rej_all = [], [], [], [], []
        for _ in range(args.prompts_per_iter):
            q, gold = problems[pi % len(problems)]; pi += 1
            grp = []
            for _g in range(args.group_size):
                samples, reward, correct, match, cost, m = rollout(
                    tiers, policy, q, gold, lat, args.lam, args.r_correct, args.match_weight)
                grp.append((samples, reward))
                r_all.append(reward); match_all.append(match)
                correct_all.append(correct); cost_all.append(cost); rej_all.append(m.rejection_rate)
            rs = torch.tensor([r for _, r in grp])
            mean, std = rs.mean(), rs.std(unbiased=False)
            for samples, reward in grp:
                adv = float((reward - mean) / (std + 1e-6))   # group-relative advantage
                for s in samples:
                    feats_buf.append(s.feats); act_buf.append(s.action); adv_buf.append(adv)
        if not feats_buf:
            continue

        Fb = torch.stack(feats_buf).to(device)
        Ab = torch.tensor(act_buf, device=device)
        Adv = torch.tensor(adv_buf, device=device, dtype=torch.float32)
        with torch.no_grad():
            old_logp = F.log_softmax(policy(Fb), dim=-1)[torch.arange(len(Ab)), Ab]
            ref_logp = F.log_softmax(ref(Fb), dim=-1)

        n = len(Ab)
        kl = ent = pg = clip_frac = gnorm = 0.0
        for _ep in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for k in range(0, n, args.minibatch):
                idx = perm[k:k + args.minibatch]
                logp_full = F.log_softmax(policy(Fb[idx]), dim=-1)
                new_logp = logp_full[torch.arange(len(idx), device=device), Ab[idx]]
                ratio = torch.exp(new_logp - old_logp[idx])
                a = Adv[idx]
                pg_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                p_new = logp_full.exp()
                kl_t = (p_new * (logp_full - ref_logp[idx])).sum(-1).mean()
                ent_t = -(p_new * logp_full).sum(-1).mean()
                loss = pg_loss + args.kl_coef * kl_t - args.entropy * ent_t
                opt.zero_grad(); loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
                opt.step()
                kl, ent, pg = float(kl_t), float(ent_t), float(pg_loss)
                clip_frac = float((torch.abs(ratio - 1) > args.clip).float().mean())

        rec = {"iter": it + 1, "reward_mean": sum(r_all) / len(r_all),
               "reward_std": float(torch.tensor(r_all).std(unbiased=False)),
               "match": sum(match_all) / len(match_all), "correct": sum(correct_all) / len(correct_all),
               "cost_per_tok": sum(cost_all) / len(cost_all), "rej": sum(rej_all) / len(rej_all),
               "adv_std": float(Adv.std(unbiased=False)), "kl": kl, "entropy": ent, "pg_loss": pg,
               "clip_frac": clip_frac, "grad_norm": gnorm, "n_tokens": n, "sec": round(time.time() - t_iter, 1)}
        jl.write(json.dumps(rec) + "\n"); jl.flush()
        if (it + 1) % args.log_every == 0:
            print("iter {iter}/{T}: reward={reward_mean:.3f}(±{reward_std:.2f}) match={match:.3f} "
                  "correct={correct:.3f} cost/tok={cost_per_tok:.3f} rej={rej:.3f} kl={kl:.4f} "
                  "ent={entropy:.3f} clip={clip_frac:.2f} gn={grad_norm:.2f} {sec}s".format(
                      T=args.iters, **rec), flush=True)
        if (it + 1) % args.ckpt_every == 0:
            save_ckpt(it + 1)
            print(f"  [ckpt @ iter {it+1} -> {args.ckpt}]", flush=True)

    save_ckpt(args.iters); jl.close()
    print(f"saved GRPO policy -> {args.out}; ckpt -> {args.ckpt}; log -> {args.jsonl}")


if __name__ == "__main__":
    main()
