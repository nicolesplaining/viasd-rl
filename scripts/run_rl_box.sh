#!/bin/bash
# RL-only box: reuse an existing policy_imitation.pt, run resumable RL + bench.
# Auto-resumes from rl_ckpt.pt if present (re-run with a larger ITERS to continue).
#   env: VIASD_VERIFIER (opt), LAM, ITERS, N_TRAIN, N_EVAL, MAX_NEW, OUT_DIR, KEEP_MASK
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PYTHON=${PYTHON:-python3}
LAM=${LAM:-0.3}; ITERS=${ITERS:-80}; N_TRAIN=${N_TRAIN:-100}; N_EVAL=${N_EVAL:-150}; MAX_NEW=${MAX_NEW:-320}
OUT_DIR=${OUT_DIR:-results/local}; mkdir -p "$OUT_DIR"
IMIT=${IMIT:-$OUT_DIR/policy_imitation.pt}; RL=${RL:-$OUT_DIR/policy_rl.pt}; CKPT=${CKPT:-$OUT_DIR/rl_ckpt.pt}
KEEP_MASK=${KEEP_MASK:-}
KARG=""; [ -n "$KEEP_MASK" ] && KARG="--keep_mask $KEEP_MASK"

echo "=== RL box: lam=$LAM iters=$ITERS verifier=${VIASD_VERIFIER:-Qwen2.5-14B} ==="; date
"$PYTHON" -m viasd.train_rl --init "$IMIT" --out "$RL" --ckpt "$CKPT" \
    --iters "$ITERS" --batch 4 --lam "$LAM" --r_correct 0.5 \
    --max_new "$MAX_NEW" --n_train "$N_TRAIN" --ckpt_every 10 --log_every 1 $KARG

echo "=== BENCHMARK ==="; "$PYTHON" scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW" \
    --policy_imitation "$IMIT" --policy_rl "$RL" $KARG
echo "=== DONE ==="; date
