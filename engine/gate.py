# Vectorized gate: the 12 q-free features (viasd/features.py) computed for all gamma draft
# positions at once on-GPU, then the trained GatingPolicy MLP -> packed [gamma, 3] int32
# (action, drafted_token, regen_token) pulled to host in ONE sync per block.
# Feature math must match viasd.features.make_features exactly (unit test in parity.py).
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.append(str(REPO))
from viasd.policy import GatingPolicy, load_policy  # noqa: E402


def load_gate(policy_path, device="cuda") -> GatingPolicy:
    pol = load_policy(str(policy_path), device=device)
    return pol.float().to(device).eval()


@torch.no_grad()
def block_features(p_logits: torch.Tensor, qp_logits: torch.Tensor, v: torch.Tensor,
                   T: int, gamma: int, max_len: int) -> torch.Tensor:
    """p_logits, qp_logits: [gamma, V] (common-vocab-sliced). v: [gamma] drafted ids.
    T = committed sequence length at block start (ids.shape[1] in the reference);
    the reference's seq_len at position j is T + j (ids grows per emitted token).
    Returns [gamma, 12] float32 in FEATURE_NAMES order."""
    g = p_logits.shape[0]
    pl = p_logits.float()
    ql = qp_logits.float()
    p = F.softmax(pl, dim=-1)
    qp = F.softmax(ql, dim=-1)
    lp = F.log_softmax(pl, dim=-1)
    lqp = F.log_softmax(ql, dim=-1)

    p_top = p.topk(2, dim=-1).values                    # [g, 2]
    p_top1, p_top2 = p_top[:, 0], p_top[:, 1]
    p_ent = -(p * lp).sum(-1)

    qp_top1, qp_arg = qp.max(dim=-1)
    qp_ent = -(qp * lqp).sum(-1)

    ar = torch.arange(g, device=p.device)
    p_v = p[ar, v]
    qp_v = qp[ar, v]
    ratio = qp_v / (qp_top1 + 1e-9)
    agree = (qp_arg == v).float()
    kl = (p * (lp - lqp)).sum(-1).clamp(min=-20.0, max=20.0)

    jnorm = ar.float() / max(gamma, 1)
    lennorm = ((T + ar.float()) / max(max_len, 1)).clamp(max=1.0)

    return torch.stack([p_v, p_top1, p_top1 - p_top2, p_ent,
                        qp_v, qp_top1, qp_ent, ratio, agree, kl,
                        jnorm, lennorm], dim=-1)


@torch.no_grad()
def gate_block(policy: GatingPolicy, p_logits, qp_logits, v, T, gamma, max_len):
    """Returns host int list-of-rows [[action, v_j, regen_tok_j] x gamma] -- ONE device sync."""
    feats = block_features(p_logits, qp_logits, v, T, gamma, max_len)   # [g,12] fp32
    actions = policy.net(feats).argmax(dim=-1)                          # [g]
    regen = qp_logits.argmax(dim=-1)                                    # [g]
    packed = torch.stack([actions.to(torch.int32), v.to(torch.int32),
                          regen.to(torch.int32)], dim=-1)
    return packed.cpu().tolist()                                        # the single sync
