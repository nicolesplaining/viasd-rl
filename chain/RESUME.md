# ViaSD × KnapSpec multi-component chain — reproduction & resume kit

**Goal:** chain Nicole's ViaSD optimizations with KnapSpec into one system and show each link
compounds, measured in *her* `bench.py` harness (metrics: q/tok, spd_cor, spd_bw, acc).
We do **not** recompute her standalone numbers — we add KnapSpec *alongside* ViaSD.

## What's in this kit (committed; survives box termination)
- `masks/dimr_mask.json` — Nicole's DIMR (KL-optimized) q′ mask, keep 26/48.
- `masks/knapspec_qprime_natural.json` — **KnapSpec-decided q′**, KnapSpec's own TPT operating point (keep 32/48).
- `masks/knapspec_qprime_match26.json` — KnapSpec-decided q′, budget-matched to DIMR (keep 26/48).
- `gen_knapspec_qprime.py` — regenerates the KnapSpec-q′ masks by running the **official KnapspecGenerator DP** on GSM8K calibration (reuses `~/work/knapspec`).
- `run_viasd_bench.sh` — per-GPU launcher for her `scripts/bench.py`. Args: `NEVAL MAXNEW CELLS`.
- `logs/` — raw bench logs from each cell (the results).
- `../scripts/aggregate_chain.py` — parses logs → `chain_summary.md` compounding table.

## The matrix (12 cells) — n_eval=30, max_new=256 run on 2024-06-20
q′-selector {evenly, DIMR(KL), KnapSpec-q′(match26), KnapSpec-q′(natural)} × RL policy {GRPO, λ0.3, λ0.6}.
Each cell = Nicole's `bench.py` → {greedy_q=AR, plain_sd=vanilla-SD, via_fixed, via_rl=chained ViaSD}.
Models: drafter Qwen2.5-0.5B-Instruct → verifier Qwen2.5-14B-Instruct, GSM8K. Trained policies from
`origin/main` results/: A=λ0.3 (`policy_rl.pt`), B=λ0.6, D=GRPO (`policy_grpo.pt`).

## Resume / scale up later (e.g. n_eval=150 for full parity with Nicole)
1. **Provision** boxes (≥1 A100-80GB per ~6 cells). HF token at `~/.hf_token`.
2. **Deploy** Nicole's code + policies + masks (bundle is rebuilt from `origin/main`):
   ```bash
   # from repo root, on your laptop:
   rm -rf /tmp/vb && mkdir -p /tmp/vb
   git archive origin/main viasd scripts requirements.txt | tar -x -C /tmp/vb
   git show origin/main:results/A_reinforce_14B_lam0.3/policy_rl.pt > /tmp/vb/policy_rl_lam03.pt
   git show origin/main:results/B_reinforce_14B_lam0.6/policy_rl.pt > /tmp/vb/policy_rl_lam06.pt
   git show origin/main:results/D_grpo_and_dimr_14B/policy_grpo.pt   > /tmp/vb/policy_grpo.pt
   cp chain/masks/dimr_mask.json chain/masks/knapspec_qprime_*.json /tmp/vb/
   tar czf /tmp/vb.tgz -C /tmp/vb .
   scp -o StrictHostKeyChecking=no /tmp/vb.tgz chain/run_viasd_bench.sh ~/.hf_token ubuntu@<IP>:~/
   ssh -o StrictHostKeyChecking=no ubuntu@<IP> 'rm -rf ~/viasd && mkdir ~/viasd && tar xzf ~/vb.tgz -C ~/viasd && source ~/venv/bin/activate && pip install -q tqdm'
   ```
   (Boxes already had a venv with torch 2.2.1+cu121 / transformers 4.57.3. Do NOT let pip upgrade torch.)
3. **Relaunch at any size** — `run_viasd_bench.sh NEVAL MAXNEW CELLS`, CELL=`gpu,policy,mask|none,verifier,tag`:
   ```bash
   V=Qwen/Qwen2.5-14B-Instruct
   CELLS="1,policy_grpo.pt,dimr_mask.json,$V,dimr_grpo;2,policy_grpo.pt,knapspec_qprime_natural.json,$V,ksnat_grpo;..."
   ssh ... ubuntu@<IP> "bash ~/run_viasd_bench.sh 150 320 '$CELLS'"
   ```
4. **Regenerate KnapSpec-q′ masks** (only if changing verifier/budget): needs `~/work/knapspec` deployed too.
   `CUDA_VISIBLE_DEVICES=0 python gen_knapspec_qprime.py` → writes the two masks to `~/viasd/`.
5. **Aggregate**: pull `~/bench_logs/*.log` into a dir, `python scripts/aggregate_chain.py <dir>`.

## Deferred link (the heavier fusion, not yet run)
Replace the 0.5B drafter with a **KnapSpec self-spec draft** *inside* `viasd/decoding.py` so the whole
hierarchy (draft → q′ → q) is self-speculative from the 14B, with interval-256 re-optimization.
This is real surgery on her generator (the draft tier must become a layer-skipped verifier forward,
not the separate 0.5B model). Scope separately; the current chain uses her 0.5B drafter.

## Note on n_eval
Rate metrics (q/tok, spd_cor, spd_bw — the compounding headline) are stable at n_eval=30 (~5k tokens/cell).
Task **accuracy** at n=30 is noisy (±~17pts CI); bump to n_eval≥100 for trustworthy accuracy (Nicole used 150).
