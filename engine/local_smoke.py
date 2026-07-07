# CPU smoke test -- runs on the laptop, no GPU, no real models. Proves the engine loops'
# bookkeeping (cache positions, fixed-shape re-feeds, escalation backfill) with tiny random
# Transformers by checking TOKEN-IDENTICAL output against a naive full-recompute reference:
#   A) engine plain_sd == engine ar (algorithmic invariant of speculative decoding)
#   B) engine via_sd  == naive-reference via_sd (same models, same gate, no caches)
# fp32; output layer scaled to sharpen argmax margins (avoids near-tie flakiness).
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from model import ModelArgs, Transformer
from tiers import Engine, build_qprime
from gate import block_features, gate_block
from viasd.features import make_features
from vsd_engine import ar_generate, plain_sd_generate, via_sd_generate
from viasd.cost import CostMeter
from viasd.policy import GatingPolicy, ACCEPT, REGEN


def tiny(name_dim=64, n_layer=6, vocab=97, seed=0):
    torch.manual_seed(seed)
    cfg = ModelArgs(block_size=512, vocab_size=vocab, n_layer=n_layer, n_head=4,
                    n_local_heads=2, dim=name_dim, intermediate_size=128, norm_eps=1e-6,
                    attn_bias=True)
    m = Transformer(cfg)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.5)
    with torch.no_grad():
        m.output.weight.mul_(4.0)          # sharpen logit margins -> stable argmax
    return m.eval()


@torch.no_grad()
def full_logits(model, toks):
    """Naive reference forward: whole prefix, no incremental caching semantics relied on."""
    x = toks.view(1, -1)
    pos = torch.arange(0, x.shape[1])
    return model(x, pos)[0]                # [L, V]


@torch.no_grad()
def ref_via_sd(drafter, q, qp, policy, prompt, gamma, max_new, vocab, eos_id):
    """Nicole's via_sd_generate semantics, naive full recompute each step."""
    ids = prompt.tolist()
    P = len(ids)
    meter = CostMeter()
    max_len = P + max_new
    while len(ids) - P < max_new:
        ids_len = len(ids)
        # draft gamma greedily
        cur = list(ids)
        p_logits = []
        for _ in range(gamma):
            lg = full_logits(drafter, torch.tensor(cur))[-1, :vocab]
            p_logits.append(lg)
            cur.append(int(lg.argmax()))
        draft = cur[ids_len:]
        meter.draft_steps += gamma
        qp_full = full_logits(qp, torch.tensor(cur))[:, :vocab]
        meter.qp_forwards += 1
        q_full = None
        q_metered = False
        broke = False
        for j in range(gamma):
            pos = ids_len - 1 + j
            v = draft[j]
            # Nicole's actual feature fn (takes j explicitly) -- cross-validates the engine's
            # vectorized block_features end-to-end
            feats = make_features(p_logits[j], qp_full[pos], v, j, gamma,
                                  seq_len=len(ids), max_len=max_len)
            action = int(policy.net(feats.view(1, -1)).argmax())
            meter.drafted += 1
            if action == ACCEPT:
                tok = v
                meter.accept += 1
            elif action == REGEN:
                tok = int(qp_full[pos].argmax())
                meter.regen += 1
            else:
                if q_full is None:
                    q_full = full_logits(q, torch.tensor(cur))[:, :vocab]
                if not q_metered:
                    meter.q_forwards += 1
                    q_metered = True
                tok = int(q_full[pos].argmax())
                meter.escalate += 1
            ids.append(tok)
            meter.tokens += 1
            changed = (tok != v)
            if action != ACCEPT and changed:
                meter.changed += 1
            if tok == eos_id:
                return torch.tensor(ids), meter
            if changed:
                broke = True
                break
        if not broke:
            continue
    return torch.tensor(ids), meter


def main():
    torch.manual_seed(42)
    vocab = 97
    gamma = 5
    max_new = 48
    drafter = tiny(48, 4, vocab, seed=1)
    q = tiny(64, 8, vocab, seed=2)
    keep = [True, False, True, True, False, True, False, True]     # keep 5/8
    qp = build_qprime(q, keep)
    for m in (drafter, q, qp):
        m.setup_caches(1, 256)

    eng = Engine(drafter=drafter, q=q, qp=qp, vocab=vocab, gamma=gamma,
                 max_seq=256, device="cpu").eager()

    torch.manual_seed(7)
    policy = GatingPolicy().eval()          # random gate: exercises all 3 actions
    prompt = torch.randint(0, vocab, (1, 12))

    # A) plain_sd == ar, token-identical
    a = ar_generate(eng, prompt, max_new, CostMeter(), eos_id=-1)
    s = plain_sd_generate(eng, prompt, max_new, CostMeter(), eos_id=-1)
    assert torch.equal(a, s), f"plain_sd != ar\n ar={a.tolist()}\n sd={s.tolist()}"
    print(f"[A] plain_sd == ar token-identical ({a.shape[0]-12} tokens)  PASS")

    # B) engine via_sd == naive reference via_sd (multiple seeds => different routing mixes)
    for seed in range(6):
        torch.manual_seed(100 + seed)
        policy = GatingPolicy().eval()
        with torch.no_grad():                      # sharpen action margins: random MLPs are
            policy.net[-1].weight.mul_(8.0)        # near-uniform -> tiny numeric diffs between
            policy.net[-1].bias.mul_(8.0)          # cached and full-recompute forwards flip
                                                   # actions; a trained gate is decisive.
        em = CostMeter()
        e = via_sd_generate(eng, prompt, max_new, em, eos_id=-1, policy=policy)
        r, rm = ref_via_sd(drafter, q, qp, policy, prompt[0], gamma, max_new, vocab, eos_id=-1)
        assert torch.equal(e, r), (f"via_sd mismatch seed={seed}\n eng={e.tolist()}\n "
                                   f"ref={r.tolist()}")
        for f in ("tokens", "accept", "regen", "escalate", "q_forwards", "qp_forwards",
                  "draft_steps", "changed", "drafted"):
            assert getattr(em, f) == getattr(rm, f), \
                f"meter.{f}: engine={getattr(em,f)} ref={getattr(rm,f)} (seed={seed})"
        acc, reg, esc = em.tier_fractions()
        print(f"[B] seed={seed}: {em.tokens} toks, a/r/e={acc:.2f}/{reg:.2f}/{esc:.2f}, "
              f"q/tok={em.q_calls_per_token:.3f}  token-identical + meters match  PASS")

    # C) EOS mid-block: force with an eos likely to appear
    e = via_sd_generate(eng, prompt, max_new, CostMeter(), eos_id=int(a[-1]), policy=policy)
    print(f"[C] EOS early-return path ran (len={e.shape[0]})  PASS")

    # D) mid-block escalation AFTER accepts: covers the commit-flush + backfill-at-j>0 path
    #    (accept j<2, escalate j>=2 -- decided purely from the jnorm feature)
    class JPolicy:
        class _Net:
            def __call__(self, feats):
                feats = feats.view(-1, 12)                       # accept 1-D or [g,12]
                j = (feats[..., 10] * 5).round().long()          # jnorm * gamma
                out = torch.full((feats.shape[0], 3), -10.0)
                out[j < 2, ACCEPT] = 10.0
                out[j >= 2, 2] = 10.0                            # ESCALATE
                return out
        net = _Net()
    jp = JPolicy()
    em = CostMeter()
    e = via_sd_generate(eng, prompt, max_new, em, eos_id=-1, policy=jp)
    r, rm = ref_via_sd(drafter, q, qp, jp, prompt[0], gamma, max_new, vocab, eos_id=-1)
    assert torch.equal(e, r), f"jpolicy mismatch\n eng={e.tolist()}\n ref={r.tolist()}"
    assert em.escalate == rm.escalate and em.accept == rm.accept and em.tokens == rm.tokens
    print(f"[D] escalate@j>=2 after accepts: {em.tokens} toks, esc={em.escalate}, "
          f"flush+backfill path token-identical  PASS")
    print("\nALL LOCAL SMOKE TESTS PASS")


if __name__ == "__main__":
    main()
