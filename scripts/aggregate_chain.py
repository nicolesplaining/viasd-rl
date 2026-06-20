#!/usr/bin/env python3
# aggregate_chain.py <logdir> — parse Nicole's bench.py table from each cell log and assemble the
# multi-component "compounding" matrix. Each cell log holds one table:
#   method  acc  rej  accept  escal  q/tok  tok/q  spd  spd_cor  spd_bw
# Cells are tagged <qprime>_<policy>:  qprime in {even, dimr, ksnat, ksm26}, policy in {grpo,lam03,lam06}.
# greedy_q = AR (1.0x ref); plain_sd = vanilla SD; via_rl = the chained ViaSD (RL routing + that q').
import glob, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "chain_logs"
METHODS = ["greedy_q", "plain_sd", "via_fixed", "via_rl"]
QP = {"even": "evenly", "dimr": "DIMR(KL)", "ksnat": "KnapSpec-q'(nat,k32)", "ksm26": "KnapSpec-q'(m26)"}
QP_ORDER = ["even", "dimr", "ksm26", "ksnat"]
POL = {"grpo": "GRPO", "lam03": "λ0.3", "lam06": "λ0.6"}
POL_ORDER = ["grpo", "lam03", "lam06"]


def parse_log(path):
    rows = {}
    for ln in open(path, errors="ignore"):
        parts = ln.split()
        if parts and parts[0] in METHODS and len(parts) >= 10:
            try:
                rows[parts[0]] = {
                    "acc": float(parts[1]), "rej": float(parts[2]),
                    "accept": float(parts[3]), "escal": float(parts[4]),
                    "qtok": float(parts[5]), "tokq": float(parts[6]),
                    "spd": float(parts[7].rstrip("x")),
                    "spd_cor": float(parts[8].rstrip("x")),
                    "spd_bw": float(parts[9].rstrip("x")),
                }
            except ValueError:
                pass
    return rows


def tag_of(fn):
    b = os.path.basename(fn)[:-4]
    for q in QP:
        if b.startswith(q + "_"):
            return q, b[len(q) + 1:]
    return None, None


def main():
    cells = {}  # (qp,pol) -> rows
    for f in glob.glob(os.path.join(ROOT, "*.log")):
        qp, pol = tag_of(f)
        if qp is None or pol not in POL:
            continue
        r = parse_log(f)
        if r:
            cells[(qp, pol)] = r

    out = ["# ViaSD × KnapSpec — multi-component chain (Nicole's bench.py, Qwen2.5-0.5B→14B, GSM8K)",
           "",
           "All numbers from her harness: q/tok = full-verifier calls per token (lower=better, hardware-free);",
           "spd_cor = launch-floor-removed speedup; spd_bw = bandwidth-model speedup. via_rl = chained ViaSD",
           "(RL routing + that q'-selector). greedy_q = AR (1.0x). plain_sd = vanilla SD.", ""]

    # in-cell baselines (take from any complete cell)
    base = next((r for r in cells.values() if "plain_sd" in r and "greedy_q" in r), None)
    if base:
        ps = base["plain_sd"]
        out += ["## Baselines (in-cell, q'-independent)",
                f"- **AR (greedy_q):** q/tok=1.000, spd=1.00x",
                f"- **Vanilla SD (plain_sd):** q/tok={ps['qtok']:.3f}, spd_cor={ps['spd_cor']:.2f}x, spd_bw={ps['spd_bw']:.2f}x, acc={ps['acc']:.3f}",
                ""]

    # main matrix: via_rl across q'-selector (rows) x policy (cols)
    out += ["## Chained ViaSD (via_rl) — does each link compound?", "",
            "**q/tok** (lower = fewer expensive q-calls = better):", ""]
    hdr = "| q'-selector \\ policy | " + " | ".join(POL[p] for p in POL_ORDER) + " |"
    out += [hdr, "|" + "---|" * (len(POL_ORDER) + 1)]
    for qp in QP_ORDER:
        cellvals = []
        for p in POL_ORDER:
            r = cells.get((qp, p), {})
            v = r.get("via_rl", {}).get("qtok")
            cellvals.append(f"{v:.3f}" if v is not None else "·")
        out.append(f"| {QP[qp]} | " + " | ".join(cellvals) + " |")

    for metric, label, suf in [("spd_cor", "spd_cor (floor-removed speedup, higher=better)", "x"),
                               ("spd_bw", "spd_bw (bandwidth-model speedup, higher=better)", "x"),
                               ("escal", "escalation rate (q-tier fraction)", ""),
                               ("acc", "task accuracy", "")]:
        out += ["", f"**{label}:**", "", hdr, "|" + "---|" * (len(POL_ORDER) + 1)]
        for qp in QP_ORDER:
            cv = []
            for p in POL_ORDER:
                r = cells.get((qp, p), {})
                v = r.get("via_rl", {}).get(metric)
                cv.append((f"{v:.2f}{suf}" if metric.startswith("spd") else f"{v:.3f}") if v is not None else "·")
            out.append(f"| {QP[qp]} | " + " | ".join(cv) + " |")

    # full per-cell dump
    out += ["", "## Full per-cell tables", ""]
    for qp in QP_ORDER:
        for p in POL_ORDER:
            r = cells.get((qp, p))
            if not r:
                continue
            out.append(f"### {QP[qp]} × {POL[p]}")
            out.append("| method | acc | escal | q/tok | spd | spd_cor | spd_bw |")
            out.append("|---|---|---|---|---|---|---|")
            for m in METHODS:
                if m in r:
                    x = r[m]
                    out.append(f"| {m} | {x['acc']:.3f} | {x['escal']:.3f} | {x['qtok']:.3f} | "
                               f"{x['spd']:.2f}x | {x['spd_cor']:.2f}x | {x['spd_bw']:.2f}x |")
            out.append("")

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(ROOT, "chain_summary.md"), "w").write(txt + "\n")
    print(f"\n[WROTE] {os.path.join(ROOT, 'chain_summary.md')}  ({len(cells)} cells parsed)")


if __name__ == "__main__":
    main()
