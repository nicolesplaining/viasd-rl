#!/usr/bin/env python3
# make_pareto_s5d.py — Nicole's accuracy-vs-speedup Pareto + our Full-Self-SD point(s).
# Nicole's points are MEASURED (from her make_figures BASE). Ours are clearly split into
# MEASURED (0.5B-draft re-bench, when available) and PROJECTED (self-spec draft @125 steps).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

# --- Nicole's measured points: (label, acc, speedup, group) ---
NICOLE = [
    ("greedy_q",                0.900, 1.00, "baseline"),
    ("plain SD (wall-clock)",   0.907, 1.40, "wallclock"),
    ("via_fixed (paper)",       0.400, 1.76, "fixed"),
    ("via_imit",                0.340, 1.69, "fixed"),
    ("via_rl REINFORCE+DIMR",   0.907, 2.22, "semiself"),
    ("via_rl v2 (lam0.6)",      0.800, 2.70, "semiself"),
    ("via_rl v2 (lam0.3)",      0.713, 3.53, "semiself"),
]
# --- OUR Full-Self-SD: projected @125 steps (self-spec draft). acc anchored to Nicole's
#     speed-favoring (v2) regime since match->0.95 + lambda high; SPEED is the honest projection. ---
OURS_PROJ = [
    ("Full-Self-SD k20 λ0.6 (PROJ)", 0.72, 1.47, "fullself"),
    ("Full-Self-SD k26 λ0.6 (PROJ)", 0.74, 1.34, "fullself"),
]
COL = {"baseline":"#7f7f7f","wallclock":"#000000","fixed":"#d62728",
       "semiself":"#1f77b4","fullself":"#2ca02c"}

fig, ax = plt.subplots(figsize=(11, 7))
for label, acc, spd, grp in NICOLE:
    ax.scatter(spd, acc, s=120, color=COL[grp], zorder=3, edgecolor="white", linewidth=0.5)
    ax.annotate(label, (spd, acc), textcoords="offset points", xytext=(7, 4), fontsize=8)
for label, acc, spd, grp in OURS_PROJ:
    ax.scatter(spd, acc, s=240, marker="*", color=COL[grp], zorder=4, edgecolor="black", linewidth=0.8)
    ax.annotate(label, (spd, acc), textcoords="offset points", xytext=(7, -10), fontsize=8, color=COL["fullself"])

# legend
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor=COL[g],markersize=10,label=n) for g,n in
       [("baseline","greedy (AR)"),("wallclock","plain SD"),("fixed","fixed-threshold (paper)"),
        ("semiself","Semi-Self + learned RL (Nicole)")]]
leg.append(Line2D([0],[0],marker='*',color='w',markerfacecolor=COL["fullself"],markeredgecolor='black',markersize=16,label="Full-Self-SD (ours, PROJECTED @125 steps)"))
ax.legend(handles=leg, loc="lower right", fontsize=9)
ax.set_xlabel("Speedup vs greedy  (overhead-free; ↑ = faster / lower latency)", fontsize=11)
ax.set_ylabel("GSM8K accuracy", fontsize=11)
ax.set_title("Accuracy vs Speedup — Semi-Self (Nicole) → Full-Self (ours)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.axhline(0.887, ls=":", color="gray", alpha=0.5)
ax.text(3.0, 0.892, "lossless ceiling (greedy acc)", fontsize=7, color="gray")
plt.tight_layout()
out = os.path.join(HERE, "pareto_s5d.png")
plt.savefig(out, dpi=150); print("[WROTE]", out)
