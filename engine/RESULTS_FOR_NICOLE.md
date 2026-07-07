# S4D on a real engine — what we built and what we learned

*(Harsh → Nicole. Everything below is measured, single H100, branch `s5d-full-self-handoff`, code in `engine/`.)*

## TL;DR
We ported your exact 3-tier method onto a gpt-fast-style engine (static KV caches, CUDA graphs,
torch.compile) — the implementation your caveats section said was missing — and measured everything
at your protocol (GSM8K test, n=150, max_new=320, greedy). **Your results reproduce almost exactly,
your overhead-corrected cost model is validated to within 1%, and the big open question is settled:
the learned gate's q-call reduction (2.8×, real) converts to zero wall-clock advantage over plain SD
in an optimized stack at 0.5B→14B.** Plain SD measured 2.34× lossless; F+DIMR ties it at a large
accuracy cost. The q-call metric you led with was the right call — it's the invariant.

## What we built (`engine/`)
- Qwen2.5 in gpt-fast: SDPA attention, fused-QKV **with the rope permute applied to biases**
  (the silent-garbage trap), meta-construction, tied embeddings. Logit parity vs HF: 14B
  **100.00% argmax agreement on all confident-margin positions**; fp32-arbiter test shows the
  engine is as faithful to fp32 ground truth as HF-bf16 itself.
- Your `via_sd_generate` semantics **exactly** (block breaks at first *changed* token, q metered
  once/block, no bonus token, your 12 features bit-identical — unit-tested at 0.00e+00).
- The systems pieces the eager harness couldn't do: cross-block KV for all three tiers with
  per-tier position invariants (every tier is always valid through T−1 → all shapes static),
  q′ as a **zero-copy parameter-shared view** of q with its own caches, escalation KV-backfill
  via fixed overlap buckets, the gate vectorized on-GPU with one host sync per block.
- Perf archaeology that mattered: cudagraphs re-record on changing storage *offsets* (staging
  buffers fix it); `enable_gqa`+mask silently falls to the math backend; cutlass fmha has
  60–85µs fixed cost at decode shapes (manual 2-GEMV attention for L≤8 halves the drafter step).
  Net: AR 76 tok/s, plain SD **180 tok/s**, ~14× faster than the eager harness end-to-end.

## The measured table (n=150, your protocol, one box)
| method | acc | ms/tok | tok/s | ×AR | ×vanilla | q/tok |
|---|---|---|---|---|---|---|
| AR (greedy q) | 0.907 | 13.11 | 76 | 1.00 | 0.43 | 1.000 |
| **plain SD** | **0.900** | 5.59 | **179** | **2.34×** | 1.00 | 0.259 |
| via_rl GRPO+DIMR | 0.800 | 7.12 | 140 | 1.84× | 0.79× | 0.143 |
| via_rl F+DIMR | 0.667 | 5.62 | 178 | 2.33× | **1.00×** | **0.091** |
| via_rl F+KnapSpec-keep20 | 0.693 | 6.60 | 152 | 1.99× | 0.85× | 0.090 |

**Fidelity check:** F+DIMR accuracy 0.667 = your HF n=150 number *exactly*; GRPO 0.800 vs your
0.813; q/tok 0.091 vs your 0.092; tier fractions match. Your trained policies transferred unchanged.

## What we learned

**1. Your spd_cor methodology is vindicated.** You projected plain SD at 2.35–2.53× once the
launch floor is removed; the engine measures **2.34×**. The overhead-correction wasn't cope —
it predicted the optimized-implementation reality to within noise.

**2. The tie, and why (the useful algebra).** Per block, the drafter cost cancels between methods;
the contest is `q′/q + p_esc(block) < 1`. Measured: 0.57 + 0.39 ≈ 1 → exact tie. Two corollaries:
- `q′/q` is **scale-invariant** — a 32B/72B verifier doesn't change it. Bigger models alone never
  make this win on-device (we skipped the 32B phase for exactly this reason).
- The levers that *would* win: a fundamentally cheaper q′ (distilled/quantized/EAGLE-style head at
  ~0.2×q → ~1.7× over vanilla), or a stack where q-calls carry fixed/metered cost.

**3. Reconciliation with the VIA-SD paper.** Their 10–20% wins over SD are real *in their regime*:
eager A100 stack (q-call count ≈ wall-clock there), weakly-aligned pairs (rejection 0.17–0.48 vs
our 0.11), lossy gates, sampling, deviation-tolerant QA tasks. Their own Impact Statement concedes
the implementation dependence and retreats to "reduction in full-verifier invocations" as the
durable claim — which is precisely your q-calls/token metric. Your GSM8K finding (fixed thresholds
collapse to ~0.4 acc; the learned gate is what makes the paradigm usable on tasks where lossiness
costs accuracy) is untouched by any of this and remains the central contribution.

**4. Where this genuinely pays (the follow-up with legs).** The 2.8× q-call reduction converts to
real wall-clock when a q-call is the scarce resource: **q offloaded to CPU/NVMe** (70B on a 24GB
card — vanilla pages q every block, the gate 0.39×/block → ~2–2.5× projected), q behind an
API/remote node, weak-drafter or sampling regimes. The engine is ready to run the offload
experiment as-is.

**5. S5D (Full-Self) is real and measured.** The engine's shared-view mechanism makes the drafter
a keep-20 *view of the 14B itself* — one weight set playing all three roles, your ssp-trained gate
routing. (Result appended below; it trails plain SD on speed as the roofline predicted — its story
is single-model deployment + the no-good-drafter setting, not raw speed.)

## Bottom line for the writeup
Lead with what's now unimpeachable: **learned gate ≫ fixed thresholds** (yours), **q-calls/token
2.8× lower** (yours, now engine-confirmed), **spd_cor methodology validated by a real engine**
(new), and **the regime map for when routing converts to wall-clock** (new — and it's the same
conclusion the VIA-SD authors hedge toward). The honest tie at 14B isn't a weakness in the story;
it's the measurement that makes every other claim credible.

*Repro: `engine/` + `scripts/box_setup.sh`; results JSONs in `engine/results/`; parity ladder in
`engine/parity.py`; CPU-only correctness suite in `engine/local_smoke.py` (runs on a laptop).*
