"""Generate figures + table for the writeup. Auto-refreshes: parses live bench logs
so re-running picks up new results, and RL curves from JSONL.

Outputs to results/figures/: speedup_bar.png, pareto.png, rl_curves.png
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

# ---- stable curated points (completed runs; see results/README.md) ----
# name, acc, spd_bw, qtok, group
# NOTE: plain SD speedup is MEASURED WALL-CLOCK (Qwen, 1.4x); all via_* speedups are the
# idealized BANDWIDTH model -- not directly comparable (real via_* wall-clock would be lower).
BASE = [
    ("greedy_q",                 0.900, 1.00,  1.000, "baseline"),
    ("plain SD (wall-clock)",    0.907, 1.40,  0.259, "wallclock"),
    ("via_fixed (paper)",        0.400, 1.76,  0.311, "fixed"),
    ("via_imit (imitation)",     0.338, 1.69,  0.288, "fixed"),
    ("via_rl REINFORCE+DIMR",    0.907, 2.22,  0.250, "learned"),
    ("via_rl v2 (lam0.3)",       0.713, 3.53,  0.104, "learned"),
]
BAR_METHODS = list(BASE)
COLORS = {"baseline": "#888888", "wallclock": "#111111", "fixed": "#d62728",
          "learned": "#1f77b4", "acc-rl": "#9467bd"}

# ---- live points parsed from bench logs (auto-refresh as runs finish) ----
# (label, logpath, group). via_rl row of each = the trained policy of that run.
LIVE = [
    ("v2 lam0.1",     os.path.join(RES, "A_lam0.1_n50", "bench_n50.log"), "learned"),
    ("v2 lam0.2",     os.path.join(RES, "B_lam0.2_n50", "bench_n50.log"), "learned"),
    ("v2 3B->14B",    os.path.join(RES, "bigger_drafter_3B_14B", "bench_n50.log"), "learned"),
    ("GRPO acc-focus",os.path.join(RES, "E_dimr_bench_and_latency", "run_grpo_acc.log"), "acc-rl"),
]

def parse_bench(path, method="via_rl"):
    """Return (acc, qtok, spd_bw) for a method row of a NEW-format bench table, or None."""
    if not os.path.exists(path):
        return None
    for line in open(path, errors="ignore"):
        p = line.split()
        if p and p[0] == method and len(p) >= 10:
            try:
                return float(p[1]), float(p[5]), float(p[9].rstrip("x"))
            except ValueError:
                continue
    return None

def collect_points():
    pts = list(BASE)
    for label, path, grp in LIVE:
        r = parse_bench(path)
        if r:
            acc, qtok, spd = r
            pts.append((label, acc, spd, qtok, grp))
    return pts

# ---------- Figure 1: speedup bar ----------
def speedup_bar():
    names = [m[0] for m in BAR_METHODS]; spd = [m[2] for m in BAR_METHODS]
    cols = [COLORS[m[4]] for m in BAR_METHODS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(names)), spd, color=cols)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs greedy (bw model; plain SD = wall-clock)")
    ax.set_title("Estimated speedup by method (overhead-free, 0.5B→14B, GSM8K)")
    for b, v in zip(bars, spd):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}×", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "speedup_bar.png"), dpi=150); plt.close()

# ---------- Figure 2: Pareto ----------
def pareto():
    pts = collect_points()
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for name, acc, spd, qtok, grp in pts:
        ax.scatter(spd, acc, s=90, color=COLORS[grp], zorder=3)
        ax.annotate(name, (spd, acc), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("Speedup vs greedy (bw model; plain SD = wall-clock)"); ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy vs speedup frontier"); ax.grid(True, alpha=0.3)
    for g, c in COLORS.items():
        ax.scatter([], [], color=c, label=g)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "pareto.png"), dpi=150); plt.close()

# ---------- Figure 3: RL curves ----------
def load_jsonl(path):
    return [json.loads(l) for l in open(path, errors="ignore")
            if l.strip().startswith("{")] if os.path.exists(path) else []

def load_reinforce(path):
    rows = []
    if os.path.exists(path):
        for l in open(path, errors="ignore"):
            m = re.search(r"iter (\d+): reward=([\d.]+) match=([\d.]+) correct=([\d.]+) cost/tok=([\d.]+) rej=([\d.]+)", l)
            if m:
                rows.append(dict(iter=int(m[1]), reward=float(m[2]), match=float(m[3]), rej=float(m[6])))
    return rows

def rl_curves():
    series = [
        ("REINFORCE",   load_reinforce(os.path.join(RES, "A_reinforce_14B_lam0.3", "run.log")), "#2ca02c"),
        ("GRPO",        load_jsonl(os.path.join(RES, "D_grpo_and_dimr_14B", "grpo_log.jsonl")), "#ff7f0e"),
        ("v2 per-token",load_jsonl(os.path.join(RES, "F_pertoken_rl_14B", "rl_pt_log.jsonl")), "#1f77b4"),
        ("GRPO acc-focus", load_jsonl(os.path.join(RES, "E_dimr_bench_and_latency", "grpo_acc_log.jsonl")), "#9467bd"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for metric, ax, title in [("reward", axes[0], "Reward (own objective)"),
                              ("match", axes[1], "Match w/ full model"),
                              ("rej", axes[2], "Rejection rate")]:
        for label, data, c in series:
            if not data:
                continue
            xs = [d["iter"] for d in data]
            key = "reward_mean" if (metric == "reward" and "reward_mean" in data[0]) else metric
            ys = [d.get(key, d.get(metric)) for d in data]
            if any(v is None for v in ys):
                continue
            ax.plot(xs, ys, label=label, color=c, lw=1.4, alpha=0.85)
        ax.set_xlabel("iteration"); ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("RL training curves (0.5B→14B, GSM8K)", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "rl_curves.png"), dpi=150, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    speedup_bar(); pareto(); rl_curves()
    print("figures refreshed:", ", ".join(sorted(os.listdir(FIG))))
    print("live points found:", [l[0] for l in LIVE if parse_bench(l[1])])
