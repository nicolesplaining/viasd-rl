#!/bin/bash
# Chained pipeline: imitation -> RL -> benchmark. Parameterized by env vars.
#   VIASD_VERIFIER (optional), LAM, N_TRAIN, EPOCHS, RL_ITERS, N_EVAL, MAX_NEW
set -e
cd ~/viasd-rl
source .venv/bin/activate

LAM=${LAM:-0.3}
N_TRAIN=${N_TRAIN:-100}
EPOCHS=${EPOCHS:-200}
RL_ITERS=${RL_ITERS:-250}
N_EVAL=${N_EVAL:-150}
MAX_NEW=${MAX_NEW:-320}

echo "=== CONFIG verifier=${VIASD_VERIFIER:-Qwen2.5-14B} lam=$LAM n_train=$N_TRAIN rl_iters=$RL_ITERS n_eval=$N_EVAL ==="
date

echo "##### STAGE 1: IMITATION #####"
python -m viasd.train_imitation --n_train "$N_TRAIN" --epochs "$EPOCHS" \
    --max_new "$MAX_NEW" --out policy_imitation.pt

echo "##### STAGE 2: RL (lam=$LAM) #####"
python -m viasd.train_rl --init policy_imitation.pt --out policy_rl.pt \
    --iters "$RL_ITERS" --batch 4 --lam "$LAM" --r_correct 0.5 \
    --max_new "$MAX_NEW" --n_train "$N_TRAIN"

echo "##### STAGE 3: BENCHMARK #####"
python scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW"

echo "=== PIPELINE DONE ==="
date
