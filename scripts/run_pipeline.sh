#!/bin/bash
# Chained pipeline: imitation -> RL -> benchmark. Parameterized by env vars.
#   VIASD_VERIFIER (optional), LAM, N_TRAIN, EPOCHS, RL_ITERS, N_EVAL, MAX_NEW, OUT_DIR
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PYTHON=${PYTHON:-python3}

LAM=${LAM:-0.3}
N_TRAIN=${N_TRAIN:-100}
EPOCHS=${EPOCHS:-200}
RL_ITERS=${RL_ITERS:-250}
N_EVAL=${N_EVAL:-150}
MAX_NEW=${MAX_NEW:-320}
OUT_DIR=${OUT_DIR:-results/local}
mkdir -p "$OUT_DIR"
IMIT="$OUT_DIR/policy_imitation.pt"
RL="$OUT_DIR/policy_rl.pt"
CKPT="$OUT_DIR/rl_ckpt.pt"
KEEP_MASK=${KEEP_MASK:-}
KARG=""; [ -n "$KEEP_MASK" ] && KARG="--keep_mask $KEEP_MASK"

echo "=== CONFIG verifier=${VIASD_VERIFIER:-Qwen2.5-14B} lam=$LAM n_train=$N_TRAIN rl_iters=$RL_ITERS n_eval=$N_EVAL keep_mask=${KEEP_MASK:-default} ==="
date

echo "##### STAGE 1: IMITATION #####"
"$PYTHON" -m viasd.train_imitation --n_train "$N_TRAIN" --epochs "$EPOCHS" \
    --max_new "$MAX_NEW" --out "$IMIT" $KARG

echo "##### STAGE 2: RL (lam=$LAM) #####"
"$PYTHON" -m viasd.train_rl --init "$IMIT" --out "$RL" --ckpt "$CKPT" \
    --iters "$RL_ITERS" --batch 4 --lam "$LAM" --r_correct 0.5 \
    --max_new "$MAX_NEW" --n_train "$N_TRAIN" $KARG

echo "##### STAGE 3: BENCHMARK #####"
"$PYTHON" scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW" \
    --policy_imitation "$IMIT" --policy_rl "$RL" $KARG

echo "=== PIPELINE DONE ==="
date
