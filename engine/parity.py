# Correctness ladder -- run IN ORDER before any stopwatch:
#   features : engine gate features == viasd.features.make_features (CPU-able, no models)
#   logits   : engine model == HF transformers, teacher-forced argmax agreement >= 99.8%
#   sd       : engine plain_sd token-identical to engine ar (within-engine, exact)
#   routing  : engine via_sd routing stats match HF-eager reference (+-3pp actions etc.)
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.append(str(REPO))
sys.path.append(str(Path(__file__).parent))


def test_features():
    from viasd.features import make_features
    from gate import block_features
    torch.manual_seed(0)
    g, V, T, max_len = 5, 1000, 137, 460
    p = torch.randn(g, V)
    qp = torch.randn(g, V)
    v = torch.randint(0, V, (g,))
    ours = block_features(p, qp, v, T, g, max_len)
    worst = 0.0
    for j in range(g):
        ref = make_features(p[j], qp[j], int(v[j]), j, g, seq_len=T + j, max_len=max_len)
        worst = max(worst, (ours[j] - ref).abs().max().item())
    print(f"[features] max|diff| = {worst:.2e}  ({'PASS' if worst <= 1e-5 else 'FAIL'})")
    assert worst <= 1e-5
    # feature order must match FEATURE_NAMES (any permutation would silently break the MLP)
    from viasd.features import FEATURE_DIM
    assert ours.shape == (g, FEATURE_DIM)


def _load_hf(model_dir, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()


def _gsm8k_prompts(n, tokenizer, device):
    from viasd.data_gsm8k import build_prompt_ids, load_gsm8k
    problems = load_gsm8k(n, split="test")
    return [(build_prompt_ids(tokenizer, q, device), gold) for q, gold in problems]


def test_logits(hf_dir, ckpt_dir, model_name, n_prompts=5, n_tok=512, device="cuda"):
    from transformers import AutoTokenizer
    from tiers import load_engine_model, COMMON_VOCAB
    tok = AutoTokenizer.from_pretrained(hf_dir)
    hf = _load_hf(hf_dir, device)
    eng = load_engine_model(model_name, ckpt_dir, device)
    with torch.device(device):
        eng.setup_caches(1, 2048)
    prompts = _gsm8k_prompts(n_prompts, tok, device)

    agree, total, worst = 0, 0, 0.0
    m_agree, m_total = 0, 0          # margin-conditioned: HF top1-top2 margin > 0.05
    for ids, _ in prompts:
        # teacher-force: HF greedy continuation, then both models score the same sequence
        with torch.no_grad():
            gen = hf.generate(ids, max_new_tokens=n_tok, do_sample=False,
                              pad_token_id=tok.eos_token_id)
            hf_logits = hf(gen).logits[0, :, :COMMON_VOCAB]
            pos = torch.arange(0, gen.shape[1], device=device)
            en_logits = eng(gen, pos)[0, :, :COMMON_VOCAB]
        P = ids.shape[1]
        hslice = hf_logits[P - 1:-1].float()
        ha = hslice.argmax(-1)
        ea = en_logits[P - 1:-1].argmax(-1)
        agree += (ha == ea).sum().item()
        total += ha.numel()
        top2 = hslice.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        conf = margin > 0.05
        m_agree += ((ha == ea) & conf).sum().item()
        m_total += conf.sum().item()
        worst = max(worst, (hslice - en_logits[P - 1:-1].float()).abs().max().item())
    rate = agree / total
    m_rate = m_agree / max(m_total, 1)
    print(f"[logits:{model_name}] argmax agreement {rate:.4%} over {total} pos "
          f"(confident-margin: {m_rate:.4%} over {m_total}), max|dlogit|={worst:.3f}  "
          f"({'PASS' if rate >= 0.998 or m_rate >= 0.999 else 'FAIL'})")
    assert rate >= 0.998 or m_rate >= 0.999, \
        "disagreements are NOT confined to near-ties -> implementation bug"


def test_sd(drafter_dir, verifier_dir, n=20, max_new=320, device="cuda"):
    from transformers import AutoTokenizer
    from tiers import build_engine
    from vsd_engine import ar_generate, plain_sd_generate
    from viasd.cost import CostMeter
    tok = AutoTokenizer.from_pretrained(verifier_dir)
    eng = build_engine(drafter_dir, verifier_dir, compile=True)
    prompts = _gsm8k_prompts(n, tok, device)
    _warmup(eng, prompts[0][0], max_new)
    bad = 0
    for ids, _ in prompts:
        a = ar_generate(eng, ids, max_new, CostMeter(), tok.eos_token_id)
        s = plain_sd_generate(eng, ids, max_new, CostMeter(), tok.eos_token_id)
        if a.shape != s.shape or not torch.equal(a, s):
            bad += 1
            k = min(a.shape[0], s.shape[0])
            div = int((a[:k] != s[:k]).nonzero()[0]) if not torch.equal(a[:k], s[:k]) else k
            print(f"  [sd] DIVERGED at pos {div} (lens {a.shape[0]} vs {s.shape[0]})")
    print(f"[sd] {n - bad}/{n} prompts token-identical  ({'PASS' if bad == 0 else 'FAIL'})")
    assert bad == 0


def test_routing(drafter_dir, verifier_dir, mask_path, policy_path, hf_drafter, hf_verifier,
                 n=30, max_new=256, device="cuda"):
    """Engine via_sd vs HF-eager reference on the same prompts: stats must match."""
    from transformers import AutoTokenizer
    from tiers import build_engine, load_keep_mask
    from vsd_engine import via_sd_generate
    from gate import load_gate
    from viasd.cost import CostMeter
    import os
    tok = AutoTokenizer.from_pretrained(verifier_dir)
    keep = load_keep_mask(mask_path)
    eng = build_engine(drafter_dir, verifier_dir, keep_mask=keep, compile=True)
    policy = load_gate(policy_path, device)
    prompts = _gsm8k_prompts(n, tok, device)
    _warmup(eng, prompts[0][0], max_new, policy=policy)

    em = CostMeter()
    for ids, _ in prompts:
        via_sd_generate(eng, ids, max_new, em, tok.eos_token_id, policy)

    # HF reference (eager, slow) -- run via Nicole's stack
    os.environ["VIASD_DRAFTER"] = hf_drafter
    os.environ["VIASD_VERIFIER"] = hf_verifier
    from viasd.config import Config
    from viasd.models import load_models
    from viasd.decoding import via_sd_generate as ref_generate, PolicyDecider
    from viasd.policy import load_policy
    cfg = Config(max_new_tokens=max_new, keep_mask_path=str(mask_path))
    tiers = load_models(cfg)
    ref_pol = load_policy(str(policy_path), device)
    rm = CostMeter()
    for ids, _ in prompts:
        ref_generate(tiers, ids, rm, PolicyDecider(ref_pol))

    def frac(m):
        return m.tier_fractions(), m.q_calls_per_token, m.tokens / max(m.qp_forwards, 1)
    (ea, er, ee), eq, etb = frac(em)
    (ra, rr, re), rq, rtb = frac(rm)
    ok = (abs(ea - ra) <= 0.03 and abs(er - rr) <= 0.03 and abs(ee - re) <= 0.03
          and abs(eq - rq) <= 0.02 and abs(etb - rtb) <= 0.3)
    print(f"[routing] engine acc/reg/esc={ea:.3f}/{er:.3f}/{ee:.3f} q/tok={eq:.3f} tok/blk={etb:.2f}")
    print(f"[routing] ref    acc/reg/esc={ra:.3f}/{rr:.3f}/{re:.3f} q/tok={rq:.3f} tok/blk={rtb:.2f}")
    print(f"[routing] {'PASS' if ok else 'FAIL'}")
    assert ok


def _warmup(eng, ids, max_new, policy=None):
    """Trigger every compile shape (incl. backfill buckets) before timing/asserting."""
    from vsd_engine import ar_generate, plain_sd_generate, via_sd_generate, BACKFILL_BUCKETS
    from viasd.cost import CostMeter
    d = eng.device
    for L in BACKFILL_BUCKETS:                      # bucket shapes (stale KV harmless: causal + rewritten)
        x = torch.zeros(1, L, dtype=torch.long, device=d)
        eng.q_fwd(x, torch.arange(0, L, device=d))
    ar_generate(eng, ids, 32, CostMeter(), -1)
    plain_sd_generate(eng, ids, 32, CostMeter(), -1)
    if policy is not None and eng.qp is not None:
        via_sd_generate(eng, ids, 32, CostMeter(), -1, policy)
    torch.cuda.synchronize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True,
                    choices=["features", "logits", "sd", "routing"])
    ap.add_argument("--hf-dir")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--model-name")
    ap.add_argument("--drafter-dir")
    ap.add_argument("--verifier-dir")
    ap.add_argument("--hf-drafter", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--hf-verifier", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--mask")
    ap.add_argument("--policy")
    a = ap.parse_args()
    if a.test == "features":
        test_features()
    elif a.test == "logits":
        test_logits(a.hf_dir, a.ckpt_dir, a.model_name)
    elif a.test == "sd":
        test_sd(a.drafter_dir, a.verifier_dir)
    elif a.test == "routing":
        test_routing(a.drafter_dir, a.verifier_dir, a.mask, a.policy,
                     a.hf_drafter, a.hf_verifier)
