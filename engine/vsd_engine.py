# Engine generate loops: ar / plain_sd / via_sd on static KV caches + compiled forwards.
# Replicates viasd/decoding.py semantics EXACTLY (block breaks at first CHANGED token; no bonus
# token; q metered once per block; EOS anywhere returns; seq_len for features = T + j).
#
# Cache-position invariant (what makes fixed shapes work): after every block, each tier has
# consumed exactly the first T-1 committed tokens -- the drafter never consumes the block's
# final committed token, and q'/q's KV at the break position was computed from the *drafted*
# (wrong) token. So: drafter catch-up is always ONE [1,1] step, and q'/q verify always re-feeds
# from position T-1 with fixed shape [1, gamma+1]. Escalation q-backfill uses overlap buckets.
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.append(str(REPO))
sys.path.append(str(Path(__file__).parent))

from viasd.cost import CostMeter          # noqa: E402
from viasd.policy import ACCEPT, REGEN, ESCALATE  # noqa: E402
from gate import gate_block               # noqa: E402

BACKFILL_BUCKETS = (8, 16, 32, 64)


class GenState:
    """Persistent device buffers reused across generate calls (cudagraph-safe copies)."""
    def __init__(self, eng):
        d, g, V = eng.device, eng.gamma, eng.vocab
        dt = eng.q.output.weight.dtype        # match model dtype -- a lower-precision buffer
        self.tokens = torch.zeros(eng.max_seq, dtype=torch.long, device=d)   # would quantize
        self.p_buf = torch.zeros(g, V, dtype=dt, device=d)                   # logits and flip
        self.qp_buf = torch.zeros(g, V, dtype=dt, device=d)                  # near-tie argmaxes
        self.draft = torch.zeros(g, dtype=torch.long, device=d)
        # fixed staging buffers: feeding views of st.tokens (whose storage OFFSET changes
        # every step) makes cudagraphs re-record per offset -> 2.5x slowdown. copy into
        # fixed-offset buffers instead.
        self.x1 = torch.zeros(1, 1, dtype=torch.long, device=d)
        self.xg = torch.zeros(1, g + 1, dtype=torch.long, device=d)


def _prefill(model, tokens, upto, device):
    """Eager prefill over tokens[0:upto] at positions 0..upto-1 (once per sequence)."""
    if upto <= 0:
        return None
    x = tokens[:upto].view(1, -1)
    pos = torch.arange(0, upto, device=device)
    return model(x, pos)


@torch.no_grad()
def ar_generate(eng, prompt_ids, max_new, meter: CostMeter, eos_id):
    """Greedy AR with the full verifier (reference greedy_q_generate)."""
    st = GenState(eng)
    P = prompt_ids.shape[1]
    st.tokens[:P] = prompt_ids[0]
    T = P
    logits = _prefill(eng.q, st.tokens, P, eng.device)           # consumes 0..P-1
    tok = int(logits[0, -1, :eng.vocab].argmax())
    meter.q1_steps += 1
    while True:
        st.tokens[T] = tok
        T += 1
        meter.tokens += 1
        if tok == eos_id or T - P >= max_new:
            break
        torch.compiler.cudagraph_mark_step_begin()
        st.x1[0, 0] = st.tokens[T - 1]
        pos = torch.arange(T - 1, T, device=eng.device)
        lg = eng.q_fwd(st.x1, pos)
        tok = int(lg[0, -1, :eng.vocab].argmax())                # sync (EOS check)
        meter.q1_steps += 1
    return st.tokens[:T].clone()


@torch.no_grad()
def _draft_gamma(eng, st, T, capture_logits: bool):
    """gamma compiled [1,1] drafter steps starting by consuming tokens[T-1].
    Fills st.draft (device); optionally st.p_buf with sliced logits per step."""
    st.x1[0, 0] = st.tokens[T - 1]
    for j in range(eng.gamma):
        torch.compiler.cudagraph_mark_step_begin()   # same-graph replay w/o step-mark re-records
        pos = torch.arange(T - 1 + j, T + j, device=eng.device)
        lg = eng.drafter_fwd(st.x1, pos)
        if capture_logits:
            st.p_buf[j].copy_(lg[0, -1, :eng.vocab])             # out of cudagraph memory
            nxt = st.p_buf[j].argmax()
        else:
            nxt = lg[0, -1, :eng.vocab].argmax().clone()
        st.draft[j] = nxt
        st.x1[0, 0] = nxt


@torch.no_grad()
def plain_sd_generate(eng, prompt_ids, max_new, meter: CostMeter, eos_id):
    """Vanilla SD (reference plain_sd_generate): accept longest argmax-matching prefix,
    correct at first mismatch, NO bonus token."""
    st = GenState(eng)
    g = eng.gamma
    P = prompt_ids.shape[1]
    st.tokens[:P] = prompt_ids[0]
    T = P
    _prefill(eng.drafter, st.tokens, P - 1, eng.device)          # consume 0..P-2 -> pos = T-1
    _prefill(eng.q, st.tokens, P - 1, eng.device)
    while T - P < max_new:
        torch.compiler.cudagraph_mark_step_begin()
        _draft_gamma(eng, st, T, capture_logits=False)
        meter.draft_steps += g
        st.xg[0, 0] = st.tokens[T - 1]
        st.xg[0, 1:] = st.draft
        pos = torch.arange(T - 1, T + g, device=eng.device)
        qlg = eng.q_fwd(st.xg, pos)
        meter.q_forwards += 1
        qarg = qlg[0, :g, :eng.vocab].argmax(dim=-1)             # rel pos j predicts T+j
        rows = torch.stack([qarg, st.draft]).cpu()               # ONE sync per block
        commits = []
        done = False
        for j in range(g):
            q_tok, d_tok = int(rows[0, j]), int(rows[1, j])
            meter.drafted += 1
            if q_tok == d_tok:
                commits.append(d_tok)
                meter.tokens += 1
                if d_tok == eos_id:
                    done = True
                    break
            else:
                commits.append(q_tok)                            # correction
                meter.tokens += 1
                meter.changed += 1
                if q_tok == eos_id:
                    done = True
                break
        st.tokens[T:T + len(commits)] = torch.tensor(commits, device=eng.device)
        T += len(commits)
        if done:
            break
    return st.tokens[:T].clone()


def _q_backfill(eng, st, q_pos, target):
    """Bring q's cache up to consumed-through target-1 and return q's argmax prediction
    for position `target`. Overlap buckets keep shapes fixed (recompute is byte-identical)."""
    gap = target - q_pos
    if not hasattr(st, "xb"):
        st.xb = {b: torch.zeros(1, b, dtype=torch.long, device=eng.device)
                 for b in BACKFILL_BUCKETS}
    while gap > BACKFILL_BUCKETS[-1]:                            # long-gap 64-chunks
        L = BACKFILL_BUCKETS[-1]
        st.xb[L][0] = st.tokens[q_pos:q_pos + L]
        torch.compiler.cudagraph_mark_step_begin()
        pos = torch.arange(q_pos, q_pos + L, device=eng.device)
        eng.q_fwd(st.xb[L], pos)
        q_pos += L
        gap = target - q_pos
    L = next(b for b in BACKFILL_BUCKETS if b >= gap)
    s = target - L                                               # overlap into consumed region: harmless
    st.xb[L][0] = st.tokens[s:target]
    torch.compiler.cudagraph_mark_step_begin()
    pos = torch.arange(s, target, device=eng.device)
    lg = eng.q_fwd(st.xb[L], pos)
    tok = int(lg[0, -1, :eng.vocab].argmax())                    # sync (escalation blocks only)
    return tok, target


@torch.no_grad()
def via_sd_generate(eng, prompt_ids, max_new, meter: CostMeter, eos_id, policy):
    """3-tier routed decoding (reference via_sd_generate semantics)."""
    st = GenState(eng)
    g = eng.gamma
    P = prompt_ids.shape[1]
    max_len = P + max_new                                        # reference: start + max_new_tokens
    st.tokens[:P] = prompt_ids[0]
    T = P
    _prefill(eng.drafter, st.tokens, P - 1, eng.device)
    _prefill(eng.qp, st.tokens, P - 1, eng.device)
    _prefill(eng.q, st.tokens, P - 1, eng.device)
    q_pos = P - 1                                                # q consumed tokens[0..q_pos-1]
    while T - P < max_new:
        torch.compiler.cudagraph_mark_step_begin()
        _draft_gamma(eng, st, T, capture_logits=True)
        meter.draft_steps += g
        st.xg[0, 0] = st.tokens[T - 1]
        st.xg[0, 1:] = st.draft
        pos = torch.arange(T - 1, T + g, device=eng.device)
        qplg = eng.qp_fwd(st.xg, pos)
        st.qp_buf.copy_(qplg[0, :g, :eng.vocab])
        meter.qp_forwards += 1
        rows = gate_block(policy, st.p_buf, st.qp_buf, st.draft, T, g, max_len)  # ONE sync
        q_metered = False
        commits = []
        flushed = 0
        done = False
        broke = False
        for j, (a, vj, rj) in enumerate(rows):
            meter.drafted += 1
            if a == ACCEPT:
                tok = vj
                meter.accept += 1
            elif a == REGEN:
                tok = rj
                meter.regen += 1
            else:                                                # ESCALATE
                if flushed < len(commits):                       # commits are q's inputs
                    new = commits[flushed:]
                    st.tokens[T + flushed:T + flushed + len(new)] = \
                        torch.tensor(new, device=eng.device)
                    flushed = len(commits)
                if not q_metered:
                    meter.q_forwards += 1
                    q_metered = True
                tok, q_pos = _q_backfill(eng, st, q_pos, T + j)
                meter.escalate += 1
            commits.append(tok)
            meter.tokens += 1
            changed = (tok != vj)
            if a != ACCEPT and changed:
                meter.changed += 1
            if tok == eos_id:
                done = True
                break
            if changed:
                broke = True
                break
        new = commits[flushed:]
        if new:
            st.tokens[T + flushed:T + flushed + len(new)] = torch.tensor(new, device=eng.device)
        T += len(commits)
        if done:
            break
    return st.tokens[:T].clone()


def timed(fn, *args, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0
