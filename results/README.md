# Results

Experiments for the RL-learned per-token gating policy on top of VIA-SD
(drafter Qwen2.5-0.5B-Instruct, verifier Qwen2.5-14B-Instruct unless noted; GSM8K).
Each subfolder is one GPU/experiment. Some runs were still in progress at collection
time (marked ⏳) — re-pull to refresh.

## Folders & files
- `local/` — default scratch/output folder for new local runs.
- `dimr_14b/` — standalone DIMR search artifacts moved out of the repo root:
  `dimr.log`, `dimr_mask.json`, and `dimr_mask_validation.json`.
- `A_reinforce_14B_lam0.3/` — REINFORCE policy, λ=0.3.
  `run.log` (imitation→RL→bench), `sweep.log` (via_fixed threshold sweep, 9 configs),
  `ctx_sweep.log` (context-length / overhead sweep), `policy_{imitation,rl}.pt`.
- `B_reinforce_14B_lam0.6/` — REINFORCE, λ=0.6 (speed-favoring). `run.log`,
  `bench_clean.log` (bench with overhead-corrected + bandwidth speedups ⏳).
- `C_reinforce_3B_lam0.3/` — capacity-gap ablation, 0.5B→**3B** verifier ⏳.
- `D_grpo_and_dimr_14B/` — `run_rl.log` (early REINFORCE), `run_grpo.log` + `grpo_log.jsonl`
  (GRPO + bench), `dimr.log` + `dimr_mask.json` (DIMR layer-mask search),
  `run_dimr_pipeline.log` (faithful retrain on DIMR mask ⏳), checkpoints.
- `E_dimr_bench_and_latency/` — `lat.log` (eager vs torch.compile latency),
  `bench_dimr.log` (bench with DIMR-optimized q′).
- `F_pertoken_rl_14B/` — v2 RL: per-token dense reward + corrected-latency cost.
  `pt_sanity.log`, `rl_pt_log.jsonl`, `run_rl_pt.log` ⏳.

## Headline metric: q-calls/token (full-verifier invocations per token; hardware-independent)

| method | accuracy | q-calls/tok | tokens/q-call |
|---|---|---|---|
| greedy_q (reference) | 0.887 | 1.000 | 1.0 |
| plain_sd (lossless SD) | 0.907 | 0.259 | 3.9 |
| via_fixed (paper, default α) | 0.380 | 0.310 | 3.2 |
| via_imit (imitation only) | 0.327 | 0.288 | 3.5 |
| **via_rl REINFORCE λ0.3 (A)** | **0.880** | 0.243 | 4.1 |
| **via_rl REINFORCE λ0.6 (B)** | 0.867 | 0.228 | 4.4 |
| **via_rl GRPO (D)** | 0.853 | **0.162** | **6.2** |

All learned variants beat plain_sd and via_fixed on q-calls/token while staying near
greedy accuracy; GRPO is most aggressive (accept ≈0.77). λ is a clean speed/quality knob.

## Speedup: overhead is an eager-execution artifact
Per-forward latency has a ~10 ms launch-overhead floor (measured: `t_q`(261 tok) ≈
`t_q1`(1 tok)). Removing it (measure-and-subtract / bandwidth model) — from `E/bench_dimr.log`:

| method | spd (measured) | spd_cor (overhead-removed) | spd_bw (bandwidth) |
|---|---|---|---|
| plain_sd | 1.06× | 2.53× | 3.32× |

i.e., plain_sd is ~2.5–3.3× once overhead is removed (matches the paper's regime).
torch.compile (fusion, no cudagraphs; `E/lat.log`) gives drafter 1.8× / verifier 1.6×;
cudagraphs are incompatible with the dynamic KV-cache loop.

## Fairness / validity notes (important)
- **via_fixed tuning doesn't rescue it.** Full sweep (`A/sweep.log`) tops out at acc 0.50
  (best ta=0.6/te=0.4) vs via_rl 0.88. Raising theta_accept *worsens* it (widens the
  regenerate band → routes more tokens to the degraded q′).
- **A better q′ (DIMR) didn't fix via_fixed either** (`E/bench_dimr.log`: 0.163 at default
  thresholds). The fixed-threshold rule is brittle to the (mask, threshold) interaction;
  the learned policy adapts to whatever q′ it's trained on.
- **Why RL ≫ imitation (0.88 vs 0.33):** mostly fixing imitation's exposure bias (RL trains
  on its own rollout states), not exotic routing — the reward is match-dominated.
- **Context sweep (`A/ctx_sweep.log`) backfired:** `measure_latencies` times the verify as a
  *non-cached* full forward, so it explodes with context (re-prefill); our impl also lacks a
  cross-block verifier cache. Main GSM8K results (~500-tok context) are unaffected; the
  bandwidth model is the reliable overhead-free estimate.
- DIMR KL: naive evenly-spaced mask = 6.22, DIMR-optimized = 3.20 (~2× better approximation).
