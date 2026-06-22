# viasd-rl

This repo implements **S4D: Self-Taught Semi-Self Speculative Decoding**.

S4D replaces VIA-SD's fixed confidence thresholds with a learned per-token gate.
For each drafted token, the gate chooses one of three actions:

- `accept`: keep the drafter token
- `regenerate`: use a slim verifier `q'`
- `escalate`: call the full verifier `q`

The gate is a small MLP over features that do not require the full verifier:
drafter and slim-verifier confidence, agreement, KL, and draft position. The full
verifier is used during training to compute labels and rewards, but not to make
routing decisions at inference.

## Details

The writeup is in [writeup/writeup.tex](writeup/writeup.tex), with a compiled PDF
at [writeup/writeup.pdf](writeup/writeup.pdf).

Main pieces:

- Reimplemented hierarchical speculative decoding with a drafter, slim verifier,
  and full verifier.
- Built the slim verifier by skipping layers of the target model, optionally using
  DIMR to search for a better layer mask.
- Trained a learned routing policy with imitation learning, then refined it with
  RL.
- Compared REINFORCE, GRPO, and per-token reward variants.
- Evaluated on GSM8K with Qwen2.5 0.5B as drafter and Qwen2.5 14B as verifier.

Headline result from the writeup: the learned gate reaches about `0.88-0.91`
GSM8K accuracy where the fixed-threshold baseline is around `0.21-0.40`, while
using fewer full-verifier calls per token. REINFORCE + DIMR matches standard
speculative decoding accuracy (`0.907`) at roughly four times fewer full-model
calls, and the per-token variant reaches the highest speedup at lower accuracy.

## Setup

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

Model names default to:

- drafter: `Qwen/Qwen2.5-0.5B-Instruct`
- verifier: `Qwen/Qwen2.5-14B-Instruct`

Override with `VIASD_DRAFTER` and `VIASD_VERIFIER`.

## Run

Quick check:

```bash
python3 scripts/smoke_test.py
python3 scripts/check_sequence_equivalence.py --n_eval 8
```

Optional DIMR mask search (paper-style random search with periodic Bayesian optimization):

```bash
python3 scripts/run_dimr.py --skip_ratio 0.45 --max_steps 60 --bo_period 10
```

Train imitation policy:

```bash
python3 -m viasd.train_imitation --n_train 120
```

Refine with RL:

```bash
python3 -m viasd.train_rl --iters 400 --lam 0.3
```

Benchmark:

```bash
python3 scripts/bench.py --n_eval 150
```

## Layout

```text
viasd/          core implementation and trainers
scripts/        command-line helpers and plotting scripts
results/        experiment logs, checkpoints, masks, and figures
results/local/  default output directory for new local runs
slides/         full slide deck and notes deck
writeup/        paper-style writeup
docs/           reference paper
```
