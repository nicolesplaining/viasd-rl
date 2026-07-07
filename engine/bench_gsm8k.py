# Measured wall-clock benchmark on the engine, protocol-matched to scripts/bench.py:
# GSM8K test, n_eval prompts, max_new=320, greedy. Reports acc, ms/tok, tok/s, x-vs-AR,
# x-vs-vanilla, q/tok, tok/block, tier fractions, and %-of-roofline. JSON + table out.
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.append(str(REPO))
sys.path.append(str(Path(__file__).parent))

from viasd.cost import CostMeter                    # noqa: E402
from viasd.data_gsm8k import build_prompt_ids, load_gsm8k  # noqa: E402
from viasd.metrics import is_correct                # noqa: E402
from tiers import build_engine, load_keep_mask      # noqa: E402
from gate import load_gate                          # noqa: E402
from vsd_engine import (ar_generate, plain_sd_generate,  # noqa: E402
                        via_sd_generate, timed)
from parity import _warmup                          # noqa: E402


def run_method(name, fn, prompts, tokenizer, max_new, eos_id):
    meter = CostMeter()
    times, toks, correct = [], 0, 0
    for i, (ids, gold) in enumerate(prompts):
        m = CostMeter()
        out, dt = timed(fn, ids, max_new, m, eos_id)
        text = tokenizer.decode(out[ids.shape[1]:], skip_special_tokens=True)
        try:
            correct += int(is_correct(text, gold))
        except (OverflowError, ValueError):
            pass    # degenerate output (e.g. huge digit string from a weak draft) = wrong
        times.append(dt)
        toks += m.tokens
        meter.merge(m)
        if (i + 1) % 25 == 0:
            print(f"  [{name}] {i+1}/{len(prompts)} acc_so_far={correct/(i+1):.3f} "
                  f"ms/tok={1e3*sum(times)/max(meter.tokens,1):.2f}", flush=True)
    ms_tok = 1e3 * sum(times) / max(meter.tokens, 1)
    a, r, e = meter.tier_fractions()
    return dict(method=name, n=len(prompts), acc=correct / len(prompts),
                ms_per_tok=ms_tok, tok_per_s=1e3 / ms_tok if ms_tok else 0,
                q_per_tok=meter.q_calls_per_token,
                tok_per_block=meter.tokens / max(meter.qp_forwards or meter.q_forwards, 1),
                accept=a, regen=r, escalate=e,
                rejection=meter.rejection_rate, total_tokens=meter.tokens,
                total_time_s=sum(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", required=True)
    ap.add_argument("--verifier-dir", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--self-draft-mask", default=None,
                    help="keep_mask json: drafter = layer-subset view of the verifier (S5D)")
    ap.add_argument("--policies", default="")   # name=path,name=path for via_rl variants
    ap.add_argument("--methods", default="ar,plain_sd,via_rl")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--out", default="bench_engine.json")
    ap.add_argument("--roofline-ar-ms", type=float, default=8.82)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.verifier_dir)
    eos = tok.eos_token_id
    device = "cuda"
    keep = load_keep_mask(a.mask) if a.mask else None
    print("[build] engine (compile on)...", flush=True)
    sdm = load_keep_mask(a.self_draft_mask) if a.self_draft_mask else None
    eng = build_engine(a.drafter_dir, a.verifier_dir, keep_mask=keep,
                       self_draft_mask=sdm, compile=True)

    problems = load_gsm8k(a.n_eval + 2, split="test")
    prompts = [(build_prompt_ids(tok, q, device), g) for q, g in problems]
    warm, prompts = prompts[:2], prompts[2:]

    policies = {}
    for spec in filter(None, a.policies.split(",")):
        nm, p = spec.split("=")
        policies[nm] = load_gate(p, device)

    print("[warmup] compiling all shapes (minutes)...", flush=True)
    t0 = time.perf_counter()
    _warmup(eng, warm[0][0], a.max_new, policy=next(iter(policies.values()), None))
    print(f"[warmup] done in {time.perf_counter()-t0:.0f}s", flush=True)

    results = []
    methods = a.methods.split(",")
    if "ar" in methods:
        results.append(run_method("ar", lambda i, m, mt, e: ar_generate(eng, i, m, mt, e),
                                  prompts, tok, a.max_new, eos))
    if "plain_sd" in methods:
        results.append(run_method("plain_sd",
                                  lambda i, m, mt, e: plain_sd_generate(eng, i, m, mt, e),
                                  prompts, tok, a.max_new, eos))
    if "via_rl" in methods:
        for nm, pol in policies.items():
            results.append(run_method(
                f"via_rl:{nm}",
                lambda i, m, mt, e, _p=pol: via_sd_generate(eng, i, m, mt, e, _p),
                prompts, tok, a.max_new, eos))

    ar_ms = next((r["ms_per_tok"] for r in results if r["method"] == "ar"), None)
    ps_ms = next((r["ms_per_tok"] for r in results if r["method"] == "plain_sd"), None)
    hdr = (f"{'method':16}{'acc':>7}{'ms/tok':>8}{'tok/s':>8}{'xAR':>7}{'xVAN':>7}"
           f"{'q/tok':>7}{'t/blk':>7}{'a/r/e':>18}{'%roof':>7}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        xar = ar_ms / r["ms_per_tok"] if ar_ms else float("nan")
        xvan = ps_ms / r["ms_per_tok"] if ps_ms else float("nan")
        roof = 100 * a.roofline_ar_ms / r["ms_per_tok"] if r["method"] == "ar" else float("nan")
        are = f"{r['accept']:.2f}/{r['regen']:.2f}/{r['escalate']:.2f}"
        print(f"{r['method']:16}{r['acc']:>7.3f}{r['ms_per_tok']:>8.2f}"
              f"{r['tok_per_s']:>8.1f}{xar:>7.2f}{xvan:>7.2f}"
              f"{r['q_per_tok']:>7.3f}{r['tok_per_block']:>7.2f}{are:>18}{roof:>7.1f}")
        r["x_ar"] = xar
        r["x_vanilla"] = xvan
    json.dump(dict(config=vars(a), results=results), open(a.out, "w"), indent=2)
    print(f"\n[WROTE] {a.out}")


if __name__ == "__main__":
    main()
