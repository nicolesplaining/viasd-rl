"""Generate figures + summary table for the writeup from collected results.

Outputs to results/figures/:
  - speedup_bar.png      : bandwidth-model speedup by method
  - pareto.png           : accuracy vs speedup (the frontier)
  - rl_curves.png        : reward / match / rejection vs iter for REINFORCE, GRPO, v2
Numbers are curated from the per-run benchmark tables (see results/README.md).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- curated benchmark numbers (accuracy, bandwidth-model speedup, q-calls/token) ----
# Primary self-consistent set: D's n=150 bench (overhead-corrected, includes all tiers).
# via_rl variants drawn from their respective benches (n noted in writeup).
# All spd_bw are REAL measured (bandwidth model) from the named bench; n noted.
# Primary self-consistent set = F's n=80 naive-mask bench (first 5).
METHODS = [
    # name,                       acc,   spd_bw, qtok,  group
    ("greedy_q",                  0.900, 1.00,  1.000, "baseline"),
    ("plain_sd (lossless)",       0.912, 3.32,  0.258, "baseline"),
    ("via_fixed (paper)",         0.400, 1.76,  0.311, "fixed"),
    ("via_imit (imitation)",      0.338, 1.69,  0.288, "fixed"),
    ("via_rl v2 (lam0.3)",        0.713, 3.53,  0.104, "learned"),
    ("via_rl REINFORCE+DIMR",     0.907, 2.22,  0.250, "learned"),
    ("via_rl v2+DIMR",            0.725, 3.47,  0.106, "learned"),
    ("via_rl v2 (32B verifier)",  0.333, 4.57,  0.070, "learned"),
]
BAR_METHODS = METHODS[:5]   # self-consistent F n=80 bench for the bar chart
COLORS = {"baseline": "#888888", "fixed": "#d62728", "learned": "#1f77b4"}

# ---------- Figure 1: speedup bar chart ----------
def speedup_bar():
    names = [m[0] for m in BAR_METHODS]
    spd = [m[2] for m in BAR_METHODS]
    cols = [COLORS[m[4]] for m in BAR_METHODS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(names)), spd, color=cols)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs greedy (bandwidth model)")
    ax.set_title("Estimated speedup by method (overhead-free, 0.5B→14B, GSM8K)")
    for b, v in zip(bars, spd):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}×", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "speedup_bar.png"), dpi=150); plt.close()

# ---------- Figure 2: Pareto (accuracy vs speedup) ----------
def pareto():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, acc, spd, qtok, grp in METHODS:
        ax.scatter(spd, acc, s=90, color=COLORS[grp], zorder=3)
        ax.annotate(name, (spd, acc), textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xlabel("Speedup vs greedy (bandwidth model)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy vs speedup frontier")
    ax.grid(True, alpha=0.3)
    # legend
    for g, c in COLORS.items():
        ax.scatter([], [], color=c, label=g)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "pareto.png"), dpi=150); plt.close()

# ---------- Figure 3: RL training curves ----------
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]

def load_reinforce(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for l in open(path):
        m = re.search(r"iter (\d+): reward=([\d.]+) match=([\d.]+) correct=([\d.]+) cost/tok=([\d.]+) rej=([\d.]+)", l)
        if m:
            rows.append(dict(iter=int(m[1]), reward=float(m[2]), match=float(m[3]),
                             correct=float(m[4]), cost_per_tok=float(m[5]), rej=float(m[6])))
    return rows

def rl_curves():
    grpo = load_jsonl(os.path.join(RES, "D_grpo_and_dimr_14B", "grpo_log.jsonl"))
    v2 = load_jsonl(os.path.join(RES, "F_pertoken_rl_14B", "rl_pt_log.jsonl"))
    rein = load_reinforce(os.path.join(RES, "A_reinforce_14B_lam0.3", "run.log"))
    series = [("REINFORCE", rein, "#2ca02c"), ("GRPO", grpo, "#ff7f0e"), ("v2 per-token", v2, "#1f77b4")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for metric, ax, title in [("reward", axes[0], "Reward (own objective)"),
                              ("match", axes[1], "Match w/ full model"),
                              ("rej", axes[2], "Rejection rate")]:
        for label, data, c in series:
            if not data:
                continue
            xs = [d["iter"] for d in data]
            key = "reward_mean" if (metric == "reward" and data and "reward_mean" in data[0]) else metric
            ys = [d.get(key, d.get(metric)) for d in data]
            ax.plot(xs, ys, label=label, color=c, lw=1.5, alpha=0.85)
        ax.set_xlabel("iteration"); ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle("RL training curves (0.5B→14B, GSM8K)", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "rl_curves.png"), dpi=150, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    speedup_bar(); pareto(); rl_curves()
    print("figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print(" -", f)
