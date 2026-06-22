# viasd-rl

Reimplementation of **VIA-SD** (Verification via Intra-Model Routing for Speculative
Decoding, arXiv:2606.12243) with a **learned per-token gating policy** that replaces
the paper's fixed `(alpha1, alpha2)` confidence thresholds.

## Idea

Speculative decoding is binary: each drafted token is either accepted from the small
drafter `p` or rejected and recomputed by the full verifier `q`. VIA-SD adds a middle
tier — a **slim-verifier `q'`** built by skipping layers of `q` — and routes each
token to one of three actions:

- **accept** the drafter token (cheap),
- **regenerate** with `q'` (medium),
- **escalate** to the full `q` (expensive).

The paper picks the route with hand-tuned thresholds on `q'`'s confidence. Here we
**learn the routing policy**: a tiny MLP over q-free features (drafter/slim-verifier
entropies, agreement, confidence ratio, KL, position). It is trained first by
**imitation** (an oracle that uses `q` at training time to pick the cost-minimal route
reproducing greedy-`q`), then refined with **REINFORCE** against a sequence-level
reward (GSM8K correctness minus a latency penalty).

## Setup (drafter -> verifier)

`Qwen2.5-0.5B-Instruct` -> `Qwen2.5-14B-Instruct`, benchmarked on a GSM8K slice.

## Install

```bash
python3 -m venv .venv --system-site-packages   # reuse system torch
source .venv/bin/activate
pip install transformers datasets numpy tqdm
```

## Run

```bash
# 0. plumbing check (2 problems, short budget)
python scripts/smoke_test.py

# 1. (optional) offline DIMR search for the layer-skip mask z*
python scripts/run_dimr.py --skip_ratio 0.45 --out dimr_mask.json

# 2. imitation pretraining of the gating policy
python -m viasd.train_imitation --n_train 120 --out policy_imitation.pt \
    [--keep_mask dimr_mask.json]

# 3. RL refinement
python -m viasd.train_rl --init policy_imitation.pt --out policy_rl.pt \
    --iters 400 --lam 1.0 [--keep_mask dimr_mask.json]

# 4. benchmark all methods on the test slice
python scripts/bench.py --n_eval 150 [--keep_mask dimr_mask.json]
```

## Metrics

SD's win is **latency, not FLOPs** (it verifies tokens it later rejects). We report:

- **acc** — GSM8K accuracy,
- **rej** — rejection rate (drafted tokens not kept as-is),
- **accept/regen/escal** — tier distribution,
- **q/tok** — full-verifier calls per token (the paper's core efficiency thesis),
- **speedup** — estimated, from per-forward latencies measured once on the GPU,
  composed by deployment-correct forward counts (a gamma-token verify is one step).

## Layout

```
viasd/
  config.py        # hyperparameters
  models.py        # loading, layer-skipped q', latency measurement
  decoding.py      # draft_block, greedy_q, plain_sd, via_sd + deciders
  features.py      # q-free policy features
  policy.py        # gating MLP
  cost.py          # forward-count meter + latency model
  dimr.py          # offline layer-mask search
  data_gsm8k.py    # prompts + loading
  metrics.py       # answer extraction / correctness
  train_imitation.py
  train_rl.py
scripts/           # smoke_test, run_dimr, bench
slides/            # full slide deck, notes deck, and PPTX builder
writeup/           # writeup source and compiled PDF
```

## Notes / caveats

- Verifier block forwards are not KV-cached across blocks (they reprocess the
  prefix), so raw wall-clock is slower than optimal — but the **reported speedup uses
  measured per-forward times x forward counts**, which is cache-invariant and
  deployment-faithful. Output tokens and routing decisions are unaffected by caching.
- The paper's `(1-alpha1)/(1-alpha2)` gate ordering is internally inconsistent with its
  reported `alpha1=0.5 > alpha2=0.3`; we use an explicit `theta_accept > theta_escalate`
  convention on the confidence ratio (see `config.py`).
