#!/usr/bin/env python3
# make_pareto_s5d.py — accuracy vs speedup (spd_bw, consistent axis).
# MEASURED: our n=150 re-bench + baselines (chain/eval_data.json). PROJECTED: Full-Self self-spec.
# Nicole's via_rl points are her reported spd_bw. (Her deck plotted plain SD at 1.4x wall-clock;
# here everything is the overhead-free bandwidth model for a fair, consistent comparison.)
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
D = json.load(open(os.path.join(HERE, "eval_data.json")))
def pt(tag, m="via_rl"): c=D[tag][m]; return c["acc"], c["spd_bw"]

# --- baselines (MEASURED, n=150) ---
ar_acc, ar_spd = pt("r_G_dimr","greedy_q")
ps_acc, ps_spd = pt("r_G_dimr","plain_sd")
# --- our MEASURED n=150 routed points ---
gk_acc, gk_spd = pt("r_G_ksm26")     # GRPO + KnapSpec-q'
gd_acc, gd_spd = pt("r_G_dimr")      # GRPO + DIMR
fk_acc, fk_spd = pt("r_F_ksm26")     # F (speed) + KnapSpec-q'

MEAS_BASE = [("greedy (AR)", ar_acc, ar_spd, "base"), ("plain SD (lossless)", ps_acc, ps_spd, "base")]
MEAS_OURS = [("GRPO + KnapSpec-q′ (ours)", gk_acc, gk_spd),
             ("GRPO + DIMR-q′ (ours)",     gd_acc, gd_spd),
             ("F + KnapSpec-q′ (ours, speed)", fk_acc, fk_spd)]
# Nicole's reported learned points (spd_bw)
NICOLE = [("via_rl REINFORCE+DIMR (Nicole)", 0.907, 2.22),
          ("via_rl v2 λ0.3 (Nicole)",        0.713, 3.53)]
# Full-Self (self-spec draft) PROJECTED @125 steps
PROJ = [("Full-Self-SD (PROJECTED)", 0.72, 1.47)]

fig, ax = plt.subplots(figsize=(11.5, 7))
for lbl, a, s, _ in MEAS_BASE:
    ax.scatter(s, a, s=130, color="#7f7f7f", zorder=3, edgecolor="white")
    ax.annotate(lbl, (s, a), textcoords="offset points", xytext=(7,5), fontsize=8)
for lbl, a, s in NICOLE:
    ax.scatter(s, a, s=120, color="#1f77b4", zorder=3, edgecolor="white")
    ax.annotate(lbl, (s, a), textcoords="offset points", xytext=(7,5), fontsize=8, color="#1f77b4")
for lbl, a, s in MEAS_OURS:
    ax.scatter(s, a, s=240, marker="*", color="#2ca02c", zorder=5, edgecolor="black", linewidth=0.8)
    ax.annotate(lbl, (s, a), textcoords="offset points", xytext=(7,-12), fontsize=8.5, color="#2ca02c", weight="bold")
for lbl, a, s in PROJ:
    ax.scatter(s, a, s=240, marker="*", facecolor="none", edgecolor="#9467bd", linewidth=1.6, zorder=5)
    ax.annotate(lbl, (s, a), textcoords="offset points", xytext=(7,5), fontsize=8, color="#9467bd")

ax.axhline(0.90, ls=":", color="gray", alpha=0.5); ax.text(1.05, 0.905, "lossless ceiling (full model)", fontsize=7, color="gray")
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor="#7f7f7f",markersize=10,label="baselines (measured)"),
       Line2D([0],[0],marker='o',color='w',markerfacecolor="#1f77b4",markersize=10,label="Nicole's learned (reported)"),
       Line2D([0],[0],marker='*',color='w',markerfacecolor="#2ca02c",markeredgecolor='black',markersize=15,label="ours: KnapSpec-q′ + RL (MEASURED n=150)"),
       Line2D([0],[0],marker='*',color='w',markerfacecolor='none',markeredgecolor="#9467bd",markersize=15,label="Full-Self-SD (PROJECTED)")]
ax.legend(handles=leg, loc="lower left", fontsize=9)
ax.set_xlabel("Speedup vs greedy  (bandwidth model, overhead-free; ↑ faster)", fontsize=11)
ax.set_ylabel("GSM8K accuracy", fontsize=11)
ax.set_title("Accuracy vs Speedup — S⁴D Pareto frontier (measured + projected)", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout(); out = os.path.join(HERE, "pareto_s5d.png"); plt.savefig(out, dpi=150); print("[WROTE]", out)
