# Full-Self Speculative Decoding (S5D) — handoff for the slide deck

> **Context for the next session:** this extends Nicole's **S4D** ("Self-Taught *Semi*-Self
> Speculative Decoding"). Read her `writeup.tex`, `slides_5min.tex`, and `results/README.md` first —
> **she is the source of truth for everything measured.** This doc adds *our* contribution on top.
> Everything below is explicitly tagged **[MEASURED]** or **[PROJECTED]**. Do not present projected
> numbers as measured.

---

## TL;DR (the one-liner)
Nicole's S4D is **Semi-Self**: the slim verifier `q'` is the target verifying its *own* draft through
a reduced copy of itself — **but the drafter is still a separate 0.5B model.** We close the loop:
**Full-Self (S5D)** makes the **drafter also a layer-skip subset of the same 14B verifier**, so **one
single model plays all three roles** — draft, `q'`, and `q` — at three KnapSpec-autotuned skip levels,
with ViaSD's RL gate routing between them. **No separate drafter. No separate slim model. One set of
weights.**

---

## The architecture (this is the "cool" part — and it's real)

```
              Nicole's S4D (Semi-Self)              Ours: S5D (Full-Self)
          ┌─────────────────────────┐          ┌─────────────────────────┐
draft  →  │ separate Qwen2.5-0.5B    │   →      │ 14B, KEEP ~8–20/48 layers│  ← KnapSpec-autotuned
          ├─────────────────────────┤          ├─────────────────────────┤
  q'   →  │ 14B, ~26/48 layers (DIMR)│          │ 14B, ~26/48 layers       │  ← KnapSpec/DIMR
          ├─────────────────────────┤          ├─────────────────────────┤
   q   →  │ 14B, full                │          │ 14B, full                │
          └─────────────────────────┘          └─────────────────────────┘
            2 models, 3 tiers                     1 model, 3 tiers  ← novelty
```

- **One weight set, three roles.** The draft, `q'`, and `q` are the *same* 14B at different layer-skip
  depths. This is *recursive* self-speculation: the model drafts for itself, slim-verifies itself, then
  full-verifies itself.
- **KnapSpec is the autotuner.** Its knapsack DP picks *which* layers each tier skips (we used its DP for
  the draft subset; DIMR or KnapSpec for `q'`). Nicole's **interval-256** tuning of KnapSpec gives **+33%**
  over the default ([MEASURED]: 1.04× vs 0.78× native).
- **ViaSD's RL gate routes.** Nicole's learned per-token policy (`policy_rl_pt.pt`) decides accept/regen/
  escalate — retrained on the self-spec draft's features (it's drafter-specific).
- **Composition:** KnapSpec (autotune skips) ∘ ViaSD (learned routing) ∘ self-spec (one model) — three
  independent ideas chained into one system.

---

## Results

### Nicole's measured Pareto (accuracy vs speedup) — **[MEASURED]**, from her `make_figures.py`
| method | acc | speedup | note |
|---|---|---|---|
| greedy_q (AR) | 0.900 | 1.00× | reference |
| plain SD | 0.907 | 1.40× (wall-clock) | lossless |
| via_fixed (paper) | 0.40 | 1.76× | hand-tuned thresholds |
| **via_rl REINFORCE+DIMR** | **0.907** | **2.22×** | lossless acc, learned gate |
| via_rl v2 (λ0.6) | 0.80 | 2.70× | learned |
| **via_rl v2 (λ0.3)** | **0.713** | **3.53×** | her speed champion |

Her headline: **lossless 0.91 acc at 4× fewer model calls, up to 3.5× faster — Pareto-dominates the paper's VIA-SD.**

### Our Full-Self-SD — convergence **[MEASURED]** + speed **[PROJECTED to ~125 steps]**
The self-spec draft gate **trains and converges**: routing fidelity `match → 0.95` in ~40 RL steps
(see `chain/ckpts/jl_*.jsonl`, plottable with her `make_figures.py:rl_curves()`). Speed extrapolated
from the real convergence trend, anchored to Nicole's convergence:

| run (draft keep × λ) | spd_cor @125 [PROJ] | match @125 [PROJ] | acc |
|---|---|---|---|
| k20 λ0.6 | **1.47×** | 0.95 | pending real re-bench¹ |
| k26 λ0.6 | 1.34× | 0.95 | pending¹ |
| k32 λ0.45 | 1.19× | 0.95 | pending¹ |

¹ acc must come from a real bench (the per-batch `correct` signal is too noisy to project). A re-bench
at Nicole's settings (n=150/max320) is the source of truth — see "In progress" below.

---

## Honest framing (do NOT overclaim — this is what makes it defensible)
- **The Full-Self win is architectural, not the Pareto.** At ~1.5× it sits *behind* Nicole's Semi-Self
  (3.5×), because a 14B-subset draft (≥2.3B active) is **structurally heavier than her 0.5B drafter** —
  it can't out-speed a tiny external model. The projection confirms this (it doesn't rescue it).
- **What it genuinely wins:** **single-model deployment** — no separate drafter to host, obtain, or align;
  the matching `match→0.95` shows it preserves fidelity; and it's the natural answer when **no good small
  drafter exists** (different model family, or a capacity gap where a 0.5B is too weak — Nicole's own
  honest-negative #(ii): "a bigger drafter is the indicated lever," and self-spec *is* that bigger,
  same-family drafter).
- **The actual speed champion (use for the speed slide) is the 0.5B-draft chain:** ViaSD-RL + DIMR/KnapSpec-q'
  → **3.70× / 3.59×** [MEASURED at n=30; re-benching at n=150/max320 for the real accuracy, expected ~0.71
  like Nicole's]. That beats vanilla SD (3.31×) on the speed axis.

### Slide narrative we'd recommend
1. **Hook:** "What if the *same model* is the drafter, the slim verifier, *and* the full verifier?"
2. **Build:** Nicole's S4D made `q'` self (Semi-Self) + learned the routing → SOTA hierarchical SD.
3. **Our step:** make the *draft* self too (Full-Self / S5D) — KnapSpec autotunes all three skip levels,
   one weight set. Show the 1-model-3-roles diagram.
4. **Evidence:** it trains (match→0.95 curve), composes KnapSpec+ViaSD+self-spec, and runs single-model.
5. **Honest trade-off:** ~1.5× (self-spec draft is heavier than a tiny external drafter) — the win is
   *architectural elegance + single-model deployment*, with the 0.5B-draft chain (3.7×) as the speed play.

---

## Figures & data in this repo
- `chain/pareto_s5d.png` — combined Pareto (Nicole's measured points + our Full-Self projected ★).
- `chain/nicole/pareto.png`, `speedup_bar.png`, `rl_curves.png` — her originals.
- `chain/projected_125.json` — exact projected stats (machine-readable).
- `chain/ckpts/jl_*.jsonl` — our real RL convergence curves (for re-plotting).
- `chain/RESULTS.md` — the full measured ablation + honest verdicts (q'-selection, latency sweep, etc.).

## In progress (next session: pull updated numbers)
- **Real re-bench at Nicole's settings (n=150, max_new=320)** running on f1 — cells `r_F_dimr`, `r_F_ksm26`,
  `r_G_dimr`, … These give the **real accuracy** (the n=30 ones were truncation/noise-deflated to ~0.47;
  expect ~0.71). Pull `~/bench_logs/r_*.log` from the box and parse the `via_rl` rows.
- To get a **real Full-Self accuracy point**, re-bench the saved self-spec policies (`chain/ckpts/ps_*.pt`)
  with `VIASD_DRAFT_MASK=knapspec_keep{8,20}.json` at n=150/max320.

## Reproduce
All code + masks + policies are in `chain/` (see `chain/RESUME.md`). Key pieces:
`viasd_models_patched.py` + `viasd_decoding_patched.py` (the self-spec-draft integration:
`draft_block_selfspec` + `VIASD_DRAFT_MASK`), `gen_knapspec_*.py` (autotuned masks),
`run_ss_pipeline.sh` (imitation→RL→bench end-to-end on the self-spec draft).
