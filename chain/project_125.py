#!/usr/bin/env python3
# project_125.py — EXTRAPOLATE each self-spec RL run's convergence to ~125 steps (Nicole's count).
# PROJECTION, not measured: fit y(t)=A+B*exp(-t/tau) to the per-iter jsonl, take A (asymptote) as
# the ~125-step value. Anchored to Nicole's reported convergence (match->~0.95). cost/tok is in
# overhead-corrected units (her method); spd_cor = 1/cost_tok. Clearly labeled PROJECTED.
import glob, json, os
import numpy as np
try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def smooth(x, w=3):
    if len(x) < w: return np.array(x, float)
    k = np.ones(w)/w
    return np.convolve(x, k, mode="valid")

def asymptote(t, y, rising):
    """Fit y=A+B*exp(-t/tau); return projected A. Fallbacks if fit is unstable."""
    t = np.array(t, float); y = np.array(y, float)
    if len(y) < 4 or not HAVE_SCIPY:
        return float(np.mean(y[-3:]))               # near-converged fallback
    try:
        B0 = y[0]-y[-1]
        p, _ = curve_fit(lambda t,A,B,tau: A+B*np.exp(-t/tau), t, y,
                         p0=[y[-1], B0, 10.0], maxfev=8000,
                         bounds=([min(y.min(),0), -5, 1],[max(y.max(),1)+1, 5, 200]))
        A = p[0]
        # sanity: asymptote shouldn't overshoot observed direction wildly
        return float(A)
    except Exception:
        return float(np.mean(y[-3:]))

rows = []
for j in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "ckpts", "jl_*.jsonl"))):
    tag = os.path.basename(j).replace("jl_","").replace(".jsonl","")
    d = [json.loads(l) for l in open(j) if l.strip()]
    if len(d) < 3:
        print(f"[{tag}] only {len(d)} steps — skip"); continue
    it = [r["iter"] for r in d]
    cost = [r["cost_per_tok"] for r in d]
    match = [r["match"] for r in d]
    corr = [r.get("correct",0) for r in d]
    cost_proj = max(asymptote(it, cost, rising=False), 0.05)
    match_proj = min(asymptote(it, match, rising=True), 0.95)     # cap at Nicole's match asymptote
    # acc: use Nicole's anchor — her via_rl acc ~= greedy(0.887) * (match/0.95-ish). Use correct-trend, floor by match scaling.
    corr_proj = float(np.clip(asymptote(it, corr, rising=True), 0, 1))
    spd_cor = 1.0/cost_proj
    rows.append((tag, len(d), cost[-1], cost_proj, spd_cor, match[-1], match_proj, corr_proj))

rows.sort(key=lambda r: -r[4])
print("\n# PROJECTED to ~125 RL steps (EXTRAPOLATION from observed trend; NOT measured)")
print("# self-spec draft = KnapSpec subset of 14B; cost/tok corrected; spd_cor=1/cost_tok\n")
print(f"{'run':10}{'obs':>5}{'cost@125':>9}{'spd_cor@125':>12}{'match@125':>10}{'acc@125':>9}")
for tag,n,cn,cp,sc,mn,mp,cr in rows:
    print(f"{tag:10}{n:>5}{cp:>9.2f}{sc:>11.2f}x{mp:>10.2f}{cr:>9.2f}")
# machine-readable for the figure/md
import json as _j
_j.dump([dict(run=t, obs=n, cost125=cp, spd_cor125=sc, match125=mp, acc125=cr)
         for t,n,cn,cp,sc,mn,mp,cr in rows],
        open(os.path.join(os.path.dirname(__file__), "projected_125.json"), "w"), indent=2)
print("\n# REFERENCE (measured): vanilla SD spd_bw 3.31x acc 0.80 ; AR 1.0x 0.80")
print("# NOTE: these are SELF-SPEC-DRAFT runs (heavy draft). The 0.5B-draft cells (f_dimr etc.)")
print("#       are being RE-BENCHED for real at n=150/max320 (use those, not projections, for the headline).")
