"""v2 RL trainer: per-token dense reward + per-token (batch-normalized) advantages,
with the cost term in OVERHEAD-CORRECTED latency units.

Motivation (vs train_rl.py / train_grpo.py): those used a single trajectory-level
reward applied to every routing decision -> crude credit assignment, high variance.
Routing decisions are mostly LOCAL, so we treat each gated token as a contextual-
bandit step:

    r_t = match_t  -  lam * cost_t  ( + r_correct * correct / T )

  match_t : 1 if the emitted token == full-verifier greedy argmax at t, else 0
  cost_t  : per-action latency in CORRECTED (overhead-free) units / t_q1:
              base = t_p1 + t_qp/gamma           (draft step + share of q' verify)
              escalate adds t_q                   (the full-q call this token triggers)
  correct : terminal GSM8K correctness, spread uniformly over the T tokens (small)

Advantage = batch-normalized r_t. Update = REINFORCE per token + entropy bonus.
Resumable checkpoints + per-iter JSONL logging.
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
from .models import corrected_latencies, load_models, measure_latencies
from .policy import ESCALATE, load_policy, save_policy


def token_cost(action, latc, gamma):
    """Per-action latency in corrected units, normalized by a full-verifier step."""
    base = latc.t_p1 + latc.t_qp / gamma          # drafter step + share of q' block verify
    if action == ESCALATE:
        base += latc.t_q                           # full-verifier call this token triggers
    return base / latc.t_q1


def rollout_full(tiers, policy, q, gold, latc, lam, r_correct, gamma):
    ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
    plen = ids.shape[1]
    samples = []
    m = CostMeter()
    out = via_sd_generate(tiers, ids, m, PolicyDecider(policy, sample=True),
                          collect=samples, record_qmatch=True)
    text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
    correct = float(is_correct(text, gold))
    n = max(len(samples), 1)
    rows = []
    for s in samples:
        match = float(s.match) if s.match >= 0 else 0.0
        r = match - lam * token_cost(s.action, latc, gamma) + r_correct * correct / n
        rows.append((s.feats, s.action, r))
    match_rate = sum(float(s.match) for s in samples if s.match >= 0) / n
    mean_cost = sum(token_cost(s.action, latc, gamma) for s in samples) / n
    return rows, correct, match_rate, mean_cost, m.rejection_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="policy_imitation.pt")
    ap.add_argument("--out", default="policy_rl_pt.pt")
    ap.add_argument("--ckpt", default="rl_pt_ckpt.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--jsonl", default="rl_pt_log.jsonl")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=6)        # rollouts pooled per update
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--r_correct", type=float, default=0.5)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", default="")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=1)
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)
    device = cfg.device
    tiers = load_models(cfg)
    print("keep_mask:", sum(tiers.keep_mask), "/", len(tiers.keep_mask), flush=True)
    latc = corrected_latencies(tiers, measure_latencies(tiers))
    print(f"corrected latencies(ms): t_p1={latc.t_p1*1e3:.2f} t_qp={latc.t_qp*1e3:.2f} "
          f"t_q={latc.t_q*1e3:.2f} t_q1={latc.t_q1*1e3:.2f}", flush=True)

    policy = load_policy(args.init, device)
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
        torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                    "pi": pi, "iter": it_done, "args": vars(args)}, args.ckpt)
        save_policy(policy, args.out)

    jl = open(args.jsonl, "a")
    for it in range(start_iter, args.iters):
        t0 = time.time()
        feats_buf, act_buf, r_buf = [], [], []
        st = {"correct": 0.0, "match": 0.0, "cost": 0.0, "rej": 0.0}
        for _ in range(args.batch):
            q, gold = problems[pi % len(problems)]; pi += 1
            rows, correct, match_rate, mean_cost, rej = rollout_full(
                tiers, policy, q, gold, latc, args.lam, args.r_correct, args.gamma)
            for f, a, r in rows:
                feats_buf.append(f); act_buf.append(a); r_buf.append(r)
            st["correct"] += correct; st["match"] += match_rate
            st["cost"] += mean_cost; st["rej"] += rej
        if not feats_buf:
            continue
        Fb = torch.stack(feats_buf).to(device)
        Ab = torch.tensor(act_buf, device=device)
        R = torch.tensor(r_buf, device=device, dtype=torch.float32)
        A = (R - R.mean()) / (R.std() + 1e-6)               # per-token, batch-normalized advantage
        logp = F.log_softmax(policy(Fb), dim=-1)
        chosen = logp[torch.arange(len(Ab), device=device), Ab]
        entropy = -(logp.exp() * logp).sum(-1).mean()
        loss = -(A * chosen).mean() - args.entropy * entropy
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        opt.step()

        b = args.batch
        rec = {"iter": it + 1, "reward": float(R.mean()), "match": st["match"] / b,
               "correct": st["correct"] / b, "cost_per_tok": st["cost"] / b,
               "rej": st["rej"] / b, "entropy": float(entropy), "grad_norm": gn,
               "n_tokens": len(Ab), "sec": round(time.time() - t0, 1)}
        jl.write(json.dumps(rec) + "\n"); jl.flush()
        if (it + 1) % args.log_every == 0:
            print("iter {iter}/{T}: tok_reward={reward:.3f} match={match:.3f} correct={correct:.3f} "
                  "cost/tok={cost_per_tok:.3f} rej={rej:.3f} ent={entropy:.3f} gn={grad_norm:.2f} "
                  "ntok={n_tokens} {sec}s".format(T=args.iters, **rec), flush=True)
        if (it + 1) % args.ckpt_every == 0:
            save_ckpt(it + 1); print(f"  [ckpt @ {it+1}]", flush=True)
    save_ckpt(args.iters); jl.close()
    print(f"saved -> {args.out}; ckpt -> {args.ckpt}; log -> {args.jsonl}")


if __name__ == "__main__":
    main()
