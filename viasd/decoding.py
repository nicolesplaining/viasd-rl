"""Decoding engines: reference greedy-q, plain speculative decoding, and the
hierarchical VIA-SD loop with a pluggable per-token decider."""
from dataclasses import dataclass

import torch

from .cost import CostMeter
from .features import make_features
from .models import lm_logits
from .policy import ACCEPT, REGEN, ESCALATE


def _cat(ids, tok):
    return torch.cat([ids, torch.tensor([[tok]], device=ids.device)], dim=1)


@torch.no_grad()
def draft_block(model, ids, gamma, vocab=None):
    """Greedily draft gamma tokens with a KV cache. Returns (tokens, per-step last-logits).
    Logits are sliced to `vocab` so drafter/verifier distributions align."""
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :vocab]
    draft, plist = [], []
    for _ in range(gamma):
        plist.append(logits[0])
        tok = int(logits[0].argmax())
        draft.append(tok)
        nxt = torch.tensor([[tok]], device=ids.device)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :vocab]
    return draft, plist


@torch.no_grad()
def reference_next_logits(tiers, ids):
    """Canonical full-verifier reference path.

    We deliberately use the same no-cache full-prefix path as block verification.
    Cached single-token decoding can take a slightly different numerical path, which
    is enough to break token equality near argmax ties.
    """
    return lm_logits(tiers.verifier, ids, None)[0, -1, :tiers.vocab]


@dataclass
class SequenceCheck:
    equal: bool
    first_mismatch: int = -1
    ref_token: int = -1
    test_token: int = -1
    ref_len: int = 0
    test_len: int = 0


def compare_generated_sequences(ref_ids, test_ids, prompt_len=0) -> SequenceCheck:
    """Compare generated suffixes from two [1, T] token tensors."""
    ref = ref_ids[0, prompt_len:].detach().cpu().tolist()
    test = test_ids[0, prompt_len:].detach().cpu().tolist()
    for i, (a, b) in enumerate(zip(ref, test)):
        if a != b:
            return SequenceCheck(False, i, a, b, len(ref), len(test))
    if len(ref) != len(test):
        return SequenceCheck(False, min(len(ref), len(test)), -1, -1, len(ref), len(test))
    return SequenceCheck(True, ref_len=len(ref), test_len=len(test))


@torch.no_grad()
def check_plain_sd_sequence_equal(tiers, ids) -> SequenceCheck:
    """Check that plain speculative decoding matches the canonical q reference."""
    prompt_len = ids.shape[1]
    ref = greedy_q_generate(tiers, ids.clone(), CostMeter())
    sd = plain_sd_generate(tiers, ids.clone(), CostMeter())
    return compare_generated_sequences(ref, sd, prompt_len)


# ----------------------------- baselines -----------------------------

@torch.no_grad()
def greedy_q_generate(tiers, ids, meter: CostMeter):
    """Reference: greedy decoding with the canonical full-verifier path."""
    cfg = tiers.cfg
    eos = tiers.tokenizer.eos_token_id
    start = ids.shape[1]
    while ids.shape[1] - start < cfg.max_new_tokens:
        logits = reference_next_logits(tiers, ids)
        tok = int(logits.argmax())
        ids = _cat(ids, tok)
        meter.tokens += 1
        meter.q1_steps += 1
        if tok == eos:
            break
    return ids


@torch.no_grad()
def plain_sd_generate(tiers, ids, meter: CostMeter):
    """Standard greedy speculative decoding: drafter proposes a block, full q
    verifies in one parallel forward, accept the longest argmax-matching prefix,
    correct at the first mismatch. Output is identical to greedy_q."""
    cfg = tiers.cfg
    eos = tiers.tokenizer.eos_token_id
    start = ids.shape[1]
    while ids.shape[1] - start < cfg.max_new_tokens:
        ids_len = ids.shape[1]
        block = min(cfg.gamma, cfg.max_new_tokens - (ids_len - start))
        draft, _ = draft_block(tiers.drafter, ids, block, tiers.vocab)
        meter.draft_steps += block
        full = torch.cat([ids, torch.tensor([draft], device=ids.device)], dim=1)
        q_logits = lm_logits(tiers.verifier, full, None)[:, :, :tiers.vocab]
        meter.q_forwards += 1
        for j in range(block):
            pos = ids_len - 1 + j
            q_tok = int(q_logits[0, pos].argmax())
            meter.drafted += 1
            if q_tok == draft[j]:
                ids = _cat(ids, draft[j])
                meter.tokens += 1
                if draft[j] == eos:
                    return ids
            else:
                ids = _cat(ids, q_tok)  # correction
                meter.tokens += 1
                meter.changed += 1
                if q_tok == eos:
                    return ids
                break
    return ids


# ----------------------------- deciders -----------------------------

class FixedThresholdDecider:
    needs_q_per_token = False
    uses_qp = True

    def __init__(self, theta_accept, theta_escalate):
        self.ta, self.te = theta_accept, theta_escalate

    def decide(self, feats, p_logits, qp_logits, q_logits, v):
        qp = torch.softmax(qp_logits, dim=-1)
        r = float(qp[v] / (qp.max() + 1e-9))
        if r >= self.ta:
            return ACCEPT
        if r >= self.te:
            return REGEN
        return ESCALATE


class OracleDecider:
    """Cost-minimal route assuming the goal is to match canonical greedy-q.

    Uses the same no-cache full-verifier reference path as the sequence-equality
    check. Training-time only.
    """
    needs_q_per_token = True
    uses_qp = True

    def decide(self, feats, p_logits, qp_logits, q_logits, v):
        q_tok = int(q_logits.argmax())
        if q_tok == v:
            return ACCEPT
        if int(qp_logits.argmax()) == q_tok:
            return REGEN
        return ESCALATE


class PolicyDecider:
    needs_q_per_token = False
    uses_qp = True

    def __init__(self, policy, sample=False):
        self.policy = policy
        self.sample = sample

    def decide(self, feats, p_logits, qp_logits, q_logits, v):
        if self.sample:
            return self.policy.act_sample(feats)
        return self.policy.act_greedy(feats)


@dataclass
class Sample:
    feats: torch.Tensor
    action: int
    match: int = -1        # 1 if emitted token == canonical q argmax at this pos
    ref_token: int = -1    # canonical q argmax token, when recorded


# ----------------------------- VIA-SD loop -----------------------------

@torch.no_grad()
def via_sd_generate(tiers, ids, meter: CostMeter, decider, collect=None, record_qmatch=False):
    """Hierarchical decoding. For each drafted token, `decider` chooses
    accept / regenerate(q') / escalate(q). The block ends at the first token
    that is changed (rejection); decoding redrafts from the new prefix.

    record_qmatch (training only): also compute the canonical reference per token
    -- WITHOUT charging it to the cost meter unless an escalation actually uses it
    -- so each collected Sample records whether the emitted token matches q's
    greedy choice. This is the dense reward signal for RL."""
    cfg = tiers.cfg
    eos = tiers.tokenizer.eos_token_id
    start = ids.shape[1]
    while ids.shape[1] - start < cfg.max_new_tokens:
        ids_len = ids.shape[1]
        block = min(cfg.gamma, cfg.max_new_tokens - (ids_len - start))
        draft, p_logits_list = draft_block(tiers.drafter, ids, block, tiers.vocab)
        meter.draft_steps += block
        full = torch.cat([ids, torch.tensor([draft], device=ids.device)], dim=1)

        qp_full = lm_logits(tiers.verifier, full, tiers.keep_mask)[:, :, :tiers.vocab]
        meter.qp_forwards += 1
        q_full = None
        q_metered = False
        use_canonical_ref = decider.needs_q_per_token or record_qmatch

        broke = False
        for j in range(block):
            pos = ids_len - 1 + j
            v = draft[j]
            p_logits_j = p_logits_list[j]
            qp_logits_j = qp_full[0, pos]
            feats = make_features(p_logits_j, qp_logits_j, v, j, cfg.gamma,
                                  ids.shape[1], start + cfg.max_new_tokens)
            q_logits_j = None
            q_charged_this_token = False
            if use_canonical_ref:
                q_logits_j = reference_next_logits(tiers, ids)
                if decider.needs_q_per_token:
                    meter.q_forwards += 1
                    q_charged_this_token = True

            action = decider.decide(feats, p_logits_j, qp_logits_j, q_logits_j, v)
            meter.drafted += 1

            if action == ACCEPT:
                tok = v
                meter.accept += 1
            elif action == REGEN:
                tok = int(qp_logits_j.argmax())
                meter.regen += 1
            else:  # ESCALATE
                if q_logits_j is None:
                    q_full = lm_logits(tiers.verifier, full, None)[:, :, :tiers.vocab]
                    q_logits_j = q_full[0, pos]
                if use_canonical_ref:
                    if not q_charged_this_token:
                        meter.q_forwards += 1
                elif not q_metered:
                    meter.q_forwards += 1
                    q_metered = True
                tok = int(q_logits_j.argmax())
                meter.escalate += 1

            if collect is not None:
                ref_tok = int(q_logits_j.argmax()) if q_logits_j is not None else -1
                match = int(tok == ref_tok) if ref_tok >= 0 else -1
                collect.append(Sample(feats.detach().cpu(), action, match, ref_tok))

            ids = _cat(ids, tok)
            meter.tokens += 1
            changed = (tok != v)
            if action != ACCEPT and changed:
                meter.changed += 1
            if tok == eos:
                return ids
            if changed:
                broke = True
                break
        if not broke:
            continue
    return ids
