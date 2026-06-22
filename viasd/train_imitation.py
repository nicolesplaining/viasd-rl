"""Imitation / supervised pretraining of the gating policy.

We roll out the OracleDecider (which uses the full verifier q to choose the
cost-minimal route that reproduces greedy-q) over training problems, logging
(q-free features, oracle action) at every drafted token, then fit the policy
with class-weighted cross-entropy. This is the strong baseline the RL stage
refines toward a sequence-level objective.
"""
import argparse

import torch
import torch.nn.functional as F

from .config import Config
from .data_gsm8k import build_prompt_ids, load_gsm8k
from .decoding import OracleDecider, via_sd_generate
from .cost import CostMeter
from .metrics import is_correct
from .models import load_models
from .paths import DEFAULT_LOCAL_RESULTS, ensure_parent_dir
from .policy import GatingPolicy, N_ACTIONS, ACTION_NAMES, save_policy


def collect_dataset(tiers, problems):
    feats, labels = [], []
    oracle = OracleDecider()
    correct = 0
    for i, (q, gold) in enumerate(problems):
        ids = build_prompt_ids(tiers.tokenizer, q, tiers.device)
        plen = ids.shape[1]
        samples = []
        m = CostMeter()
        out = via_sd_generate(tiers, ids, m, oracle, collect=samples)
        text = tiers.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        correct += int(is_correct(text, gold))
        for s in samples:
            feats.append(s.feats)
            labels.append(s.action)
        print(f"[collect {i+1}/{len(problems)}] samples={len(samples)} "
              f"acc_so_far={correct/(i+1):.3f}", flush=True)
    X = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.long)
    print(f"oracle (==greedy-q) accuracy: {correct/len(problems):.3f}")
    return X, y


def train(X, y, device, epochs=200, lr=1e-3, hidden=64):
    policy = GatingPolicy(hidden=hidden).to(device)
    counts = torch.bincount(y, minlength=N_ACTIONS).float()
    weights = (counts.sum() / (counts + 1e-6))
    weights = (weights / weights.sum() * N_ACTIONS).to(device)
    print("action counts:", {ACTION_NAMES[i]: int(counts[i]) for i in range(N_ACTIONS)})
    Xd, yd = X.to(device), y.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = Xd.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for k in range(0, n, 512):
            idx = perm[k:k + 512]
            logits = policy(Xd[idx])
            loss = F.cross_entropy(logits, yd[idx], weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                acc = (policy(Xd).argmax(-1) == yd).float().mean().item()
            print(f"epoch {ep+1}: loss={tot/n:.4f} train_action_acc={acc:.3f}", flush=True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--keep_mask", type=str, default="")
    ap.add_argument("--out", type=str, default=str(DEFAULT_LOCAL_RESULTS / "policy_imitation.pt"))
    args = ap.parse_args()

    cfg = Config(max_new_tokens=args.max_new, keep_mask_path=args.keep_mask)
    torch.manual_seed(cfg.seed)
    tiers = load_models(cfg)
    print("keep_mask:", tiers.keep_mask, flush=True)
    problems = load_gsm8k(args.n_train, split=args.split)
    X, y = collect_dataset(tiers, problems)
    data_path = args.out.replace(".pt", "_data.pt")
    ensure_parent_dir(data_path)
    torch.save({"X": X, "y": y}, data_path)
    policy = train(X, y, cfg.device, epochs=args.epochs)
    save_policy(policy, args.out)
    print(f"saved policy -> {args.out}")


if __name__ == "__main__":
    main()
