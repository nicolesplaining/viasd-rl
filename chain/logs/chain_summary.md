# ViaSD × KnapSpec — multi-component chain (Nicole's bench.py, Qwen2.5-0.5B→14B, GSM8K)

All numbers from her harness: q/tok = full-verifier calls per token (lower=better, hardware-free);
spd_cor = launch-floor-removed speedup; spd_bw = bandwidth-model speedup. via_rl = chained ViaSD
(RL routing + that q'-selector). greedy_q = AR (1.0x). plain_sd = vanilla SD.

## Baselines (in-cell, q'-independent)
- **AR (greedy_q):** q/tok=1.000, spd=1.00x
- **Vanilla SD (plain_sd):** q/tok=0.259, spd_cor=2.29x, spd_bw=3.31x, acc=0.800

## Chained ViaSD (via_rl) — does each link compound?

**q/tok** (lower = fewer expensive q-calls = better):

| q'-selector \ policy | GRPO | λ0.3 | λ0.6 |
|---|---|---|---|
| evenly | 0.162 | 0.243 | 0.253 |
| DIMR(KL) | 0.149 | 0.240 | 0.253 |
| KnapSpec-q'(m26) | 0.152 | 0.245 | 0.253 |
| KnapSpec-q'(nat,k32) | 0.157 | 0.242 | 0.254 |

**spd_cor (floor-removed speedup, higher=better):**

| q'-selector \ policy | GRPO | λ0.3 | λ0.6 |
|---|---|---|---|
| evenly | 1.70x | 1.32x | 1.36x |
| DIMR(KL) | 1.77x | 1.28x | 1.69x |
| KnapSpec-q'(m26) | 2.23x | 2.42x | 1.09x |
| KnapSpec-q'(nat,k32) | 1.10x | 1.13x | 1.13x |

**spd_bw (bandwidth-model speedup, higher=better):**

| q'-selector \ policy | GRPO | λ0.3 | λ0.6 |
|---|---|---|---|
| evenly | 2.82x | 2.26x | 2.21x |
| DIMR(KL) | 2.90x | 2.28x | 2.21x |
| KnapSpec-q'(m26) | 2.90x | 2.26x | 2.22x |
| KnapSpec-q'(nat,k32) | 2.62x | 2.13x | 2.07x |

**escalation rate (q-tier fraction):**

| q'-selector \ policy | GRPO | λ0.3 | λ0.6 |
|---|---|---|---|
| evenly | 0.239 | 0.517 | 0.583 |
| DIMR(KL) | 0.202 | 0.563 | 0.675 |
| KnapSpec-q'(m26) | 0.209 | 0.536 | 0.641 |
| KnapSpec-q'(nat,k32) | 0.228 | 0.571 | 0.683 |

**task accuracy:**

| q'-selector \ policy | GRPO | λ0.3 | λ0.6 |
|---|---|---|---|
| evenly | 0.700 | 0.700 | 0.700 |
| DIMR(KL) | 0.633 | 0.700 | 0.700 |
| KnapSpec-q'(m26) | 0.700 | 0.800 | 0.767 |
| KnapSpec-q'(nat,k32) | 0.700 | 0.733 | 0.767 |

## Full per-cell tables

### evenly × GRPO
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.97x | 1.70x | 3.31x |
| via_fixed | 0.267 | 0.686 | 0.312 | 0.61x | 1.01x | 1.74x |
| via_rl | 0.700 | 0.239 | 0.162 | 0.91x | 1.70x | 2.82x |

### evenly × λ0.3
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.96x | 1.72x | 3.31x |
| via_fixed | 0.267 | 0.686 | 0.312 | 0.61x | 1.02x | 1.74x |
| via_rl | 0.700 | 0.517 | 0.243 | 0.79x | 1.32x | 2.26x |

### evenly × λ0.6
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.96x | 1.78x | 3.31x |
| via_fixed | 0.267 | 0.686 | 0.312 | 0.61x | 1.08x | 1.74x |
| via_rl | 0.700 | 0.583 | 0.253 | 0.79x | 1.36x | 2.21x |

### DIMR(KL) × GRPO
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.94x | 1.67x | 3.31x |
| via_fixed | 0.167 | 0.379 | 0.247 | 0.65x | 1.18x | 2.00x |
| via_rl | 0.633 | 0.202 | 0.149 | 0.89x | 1.77x | 2.90x |

### DIMR(KL) × λ0.3
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.95x | 1.65x | 3.31x |
| via_fixed | 0.167 | 0.379 | 0.247 | 0.66x | 1.15x | 2.00x |
| via_rl | 0.700 | 0.563 | 0.240 | 0.79x | 1.28x | 2.28x |

### DIMR(KL) × λ0.6
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.99x | 2.29x | 3.31x |
| via_fixed | 0.167 | 0.379 | 0.247 | 0.69x | 1.57x | 2.00x |
| via_rl | 0.700 | 0.675 | 0.253 | 0.82x | 1.69x | 2.21x |

### KnapSpec-q'(m26) × GRPO
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 1.08x | 2.40x | 3.31x |
| via_fixed | 0.233 | 0.510 | 0.290 | 0.69x | 1.37x | 1.82x |
| via_rl | 0.700 | 0.209 | 0.152 | 1.00x | 2.23x | 2.90x |

### KnapSpec-q'(m26) × λ0.3
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.96x | 2.60x | 3.31x |
| via_fixed | 0.233 | 0.510 | 0.290 | 0.65x | 2.00x | 1.82x |
| via_rl | 0.800 | 0.536 | 0.245 | 0.83x | 2.42x | 2.26x |

### KnapSpec-q'(m26) × λ0.6
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.85x | 1.42x | 3.31x |
| via_fixed | 0.233 | 0.510 | 0.290 | 0.56x | 0.91x | 1.82x |
| via_rl | 0.767 | 0.641 | 0.253 | 0.70x | 1.09x | 2.22x |

### KnapSpec-q'(nat,k32) × GRPO
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.93x | 1.56x | 3.31x |
| via_fixed | 0.167 | 0.300 | 0.211 | 0.63x | 0.87x | 2.08x |
| via_rl | 0.700 | 0.228 | 0.157 | 0.78x | 1.10x | 2.62x |

### KnapSpec-q'(nat,k32) × λ0.3
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.95x | 1.65x | 3.31x |
| via_fixed | 0.167 | 0.300 | 0.211 | 0.69x | 1.14x | 2.08x |
| via_rl | 0.733 | 0.571 | 0.242 | 0.75x | 1.13x | 2.13x |

### KnapSpec-q'(nat,k32) × λ0.6
| method | acc | escal | q/tok | spd | spd_cor | spd_bw |
|---|---|---|---|---|---|---|
| greedy_q | 0.800 | 0.000 | 1.000 | 1.00x | 1.00x | 1.00x |
| plain_sd | 0.800 | 0.000 | 0.259 | 0.94x | 1.69x | 3.31x |
| via_fixed | 0.167 | 0.300 | 0.211 | 0.69x | 1.17x | 2.08x |
| via_rl | 0.767 | 0.683 | 0.254 | 0.74x | 1.13x | 2.07x |

