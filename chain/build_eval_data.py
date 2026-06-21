#!/usr/bin/env python3
# build_eval_data.py — parse ALL eval logs in chain/eval_all/ into a clean dataset + table.
import glob, json, os, re
HERE = os.path.dirname(__file__)
M = ["greedy_q", "plain_sd", "via_fixed", "via_rl"]

def n_eval(tag):
    if tag.startswith("r_"): return 150
    if tag.startswith("sm_"): return 2
    return 30

def draft(tag):
    # all completed benches use the 0.5B drafter (Semi-Self). Self-spec runs were train-only (projected).
    return "0.5B (Semi-Self)"

rows = {}
for f in sorted(glob.glob(os.path.join(HERE, "eval_all", "*.log"))):
    tag = os.path.basename(f)[:-4]
    cell = {}
    for ln in open(f, errors="ignore"):
        p = ln.split()
        if p and p[0] in M and len(p) >= 10:
            try:
                cell[p[0]] = dict(acc=float(p[1]), escal=float(p[4]), qtok=float(p[5]),
                                  spd=float(p[7].rstrip("x")), spd_cor=float(p[8].rstrip("x")),
                                  spd_bw=float(p[9].rstrip("x")))
            except ValueError: pass
    if "via_rl" in cell:
        rows[tag] = dict(n_eval=n_eval(tag), draft=draft(tag), **cell)

json.dump(rows, open(os.path.join(HERE, "eval_data.json"), "w"), indent=2)

# markdown table: the n=150 (real) cells first, then n=30
def line(tag, c):
    v = c["via_rl"]; ps = c.get("plain_sd", {})
    return (f"| {tag} | {c['n_eval']} | {v['acc']:.3f} | {v['qtok']:.3f} | {v['spd_bw']:.2f}x | "
            f"{v['spd_cor']:.2f}x | {ps.get('acc','-')} | {ps.get('spd_bw','-')} |")

out = ["# All eval data (via_rl = the routed method; plain_sd = vanilla-SD baseline in same run)",
       "", "| cell | n | acc | q/tok | spd_bw | spd_cor | plain_sd acc | plain_sd spd_bw |",
       "|---|---|---|---|---|---|---|---|"]
for tag in sorted(rows, key=lambda t: (-rows[t]["n_eval"], t)):
    out.append(line(tag, rows[tag]))
open(os.path.join(HERE, "eval_data.md"), "w").write("\n".join(out) + "\n")
print(f"parsed {len(rows)} cells -> eval_data.json + eval_data.md")
print("\n=== n=150 REAL points (via_rl) ===")
for t in sorted(rows):
    if rows[t]["n_eval"] == 150:
        v = rows[t]["via_rl"]; print(f"  {t:12} acc={v['acc']:.3f} spd_bw={v['spd_bw']:.2f}x q/tok={v['qtok']:.3f}")
