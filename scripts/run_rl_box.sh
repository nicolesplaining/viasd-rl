#!/bin/bash
# RL-only box: reuse an existing policy_imitation.pt, run resumable RL + bench.
# Auto-resumes from rl_ckpt.pt if present (re-run with a larger ITERS to continue).
#   env: VIASD_VERIFIER (opt), LAM, ITERS, N_TRAIN, N_EVAL, MAX_NEW
set -e
cd ~/viasd-rl
source .venv/bin/activate
LAM=${LAM:-0.3}; ITERS=${ITERS:-80}; N_TRAIN=${N_TRAIN:-100}; N_EVAL=${N_EVAL:-150}; MAX_NEW=${MAX_NEW:-320}

echo "=== RL box: lam=$LAM iters=$ITERS verifier=${VIASD_VERIFIER:-Qwen2.5-14B} ==="; date
python -m viasd.train_rl --init policy_imitation.pt --out policy_rl.pt --ckpt rl_ckpt.pt \
    --iters "$ITERS" --batch 4 --lam "$LAM" --r_correct 0.5 \
    --max_new "$MAX_NEW" --n_train "$N_TRAIN" --ckpt_every 10 --log_every 1

echo "=== BENCHMARK ==="; python scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW"
echo "=== DONE ==="; date
