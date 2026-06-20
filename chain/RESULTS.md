# ViaSD × KnapSpec — results & honest verdict
Qwen2.5-0.5B→14B, GSM8K. ViaSD numbers: Nicole's `bench.py` cost-model (q/tok exact;
spd_bw = overhead-free bandwidth speedup). Native-KnapSpec: its own KV-cached wall-clock harness.
**n=30** (rate metrics q/tok, spd_bw stable; accuracy ±~17pt CI — treat acc deltas as noisy).

## What we set out to test: a 3-component system
1. **KnapSpec autotuner** (knapsack layer-skip + interval tuning)
2. **ViaSD RL routing** (learned accept/regen/escalate gate)
3. **KnapSpec-DP picks q′** (instead of DIMR's KL search)  ← our novel third leg

## Component wins (each vs its OWN baseline) — VALIDATED ✅
- **ViaSD RL routing ≫ fixed-threshold gating** (Nicole's central result, reproduced): learned gate
  reaches acc 0.71–0.91 vs via_fixed ≤0.50, at lower q/tok. Robust.
- **KnapSpec interval-256 > interval-64: +33%** (native harness): 1.04× vs 0.78× measured speedup,
  entirely from cutting optimize overhead (14.0s → 3.2s/run) at equal acceptance (~0.82).

## Leg 3 (KnapSpec-DP for q′ vs DIMR) — HONEST WASH ❌
- **Whole-block, equal budget:** KnapSpec-DP ≈ DIMR (e.g. F-policy spd_bw: DIMR 3.70× vs KnapSpec 3.59×;
  q/tok 0.090 vs 0.098). DIMR is KL-optimal at fixed budget; KnapSpec's TPT-knapsack reduces to the
  same thing → a tie. KnapSpec-DP is a **calibration-free drop-in for DIMR at parity** — a convenience,
  not a win.
- **Sublayer (attn/MLP-decoupled) q′, even at 16K context:** does NOT beat DIMR. q′ forward latency (ms):
  | ctx | DIMR-block | KnapSpec sublayer (attn+mlp, matched budget) |
  |---|---|---|
  | 256 | cheaper | slower |
  | 16384 | **3107** | 3211 |
  MLP params dominate bandwidth and DIMR already drops them; attention's O(n²) compute doesn't overtake
  even at 16K. The "skip attention to win at long context" hypothesis did not pan out here.

## The Pareto bar (combination must beat BOTH components independently) — NOT MET ❌
In one metric (spd_bw vs acc, n=30):

| method | spd_bw | acc |
|---|---|---|
| **vanilla SD** | 3.31× | **0.80** |
| ViaSD-F + DIMR (combination) | 3.70× | 0.47 |
| ViaSD-F + KnapSpec-q′ | 3.59× | 0.47 |
| ViaSD-GRPO + DIMR | 2.90× | 0.63 |
| ViaSD-GRPO + KnapSpec-q′ | 2.90× | 0.70 |

**Plain lossless SD is Pareto-competitive with every ViaSD variant** at this 0.5B→14B gap — it buys
acc 0.80 at 3.31×, while ViaSD trades large accuracy for a sliver of speed. This is **Nicole's own
documented honest-negative #(i)** ("the q′ middle tier's per-block overhead doesn't pay off"), and our
data confirms it. The fused combination does **not** push the frontier past vanilla SD or past
ViaSD-alone.

## What can be honestly presented
1. ViaSD RL routing beats fixed-threshold gating (reproduced).
2. KnapSpec interval-256 beats interval-64 by +33% (validated).
3. KnapSpec-DP = DIMR at parity, **without** DIMR's KL calibration step (a usability result).
4. Documented limitation (shared with Nicole): plain SD stays Pareto-competitive at this capacity gap;
   the q′ middle tier and the KnapSpec×ViaSD fusion don't beat it on the accuracy–speed frontier.

## Not run (would be needed for a real combination claim)
- **Self-spec DRAFT** (replace the 0.5B drafter with a skipped-14B) — the leg that would earn the
  "Self-Self-Speculative" name and the only untested path to a genuine combination win. Deferred.
- Larger capacity gap / a regime where the q′ tier's overhead is justified.
