"""Cheap, q-free features for the per-token gating policy.

Crucial constraint: these must be computable WITHOUT running the full verifier q,
since the whole point is to decide whether to invoke q. They use only the drafter
distribution p and the slim-verifier distribution q'. The full verifier is used
only at training time (for oracle labels / RL reward), never as a policy input.
"""
import torch
import torch.nn.functional as F

FEATURE_NAMES = [
    "p_v",       # drafter prob of its own (drafted) token
    "p_top1",    # drafter max prob
    "p_margin",  # drafter top1 - top2
    "p_ent",     # drafter entropy (nats)
    "qp_v",      # slim-verifier prob of the drafted token
    "qp_top1",   # slim-verifier max prob (M')
    "qp_ent",    # slim-verifier entropy
    "ratio",     # q'(v) / max_u q'(u)  -- the paper's confidence ratio
    "agree",     # 1 if argmax q' == drafted token
    "kl_p_qp",   # KL(p || q')
    "jnorm",     # position within the draft block
    "lennorm",   # sequence length so far / max
]
FEATURE_DIM = len(FEATURE_NAMES)


@torch.no_grad()
def make_features(p_logits, qp_logits, v, j, gamma, seq_len, max_len, device=None):
    """p_logits, qp_logits: [V] logits at this position. v: drafted token id."""
    device = device or p_logits.device
    p = F.softmax(p_logits, dim=-1)
    qp = F.softmax(qp_logits, dim=-1)
    lp = F.log_softmax(p_logits, dim=-1)
    lqp = F.log_softmax(qp_logits, dim=-1)

    p_top = p.topk(2).values
    p_top1, p_top2 = p_top[0], p_top[1]
    p_ent = -(p * lp).sum()

    qp_top1, qp_arg = qp.max(dim=-1)
    qp_ent = -(qp * lqp).sum()

    p_v = p[v]
    qp_v = qp[v]
    ratio = qp_v / (qp_top1 + 1e-9)
    agree = (qp_arg == v).float()
    kl = (p * (lp - lqp)).sum().clamp(min=-20.0, max=20.0)

    return torch.stack([
        p_v, p_top1, p_top1 - p_top2, p_ent,
        qp_v, qp_top1, qp_ent, ratio, agree, kl,
        torch.tensor(j / max(gamma, 1), device=device),
        torch.tensor(min(seq_len / max(max_len, 1), 1.0), device=device),
    ]).to(device).float()
