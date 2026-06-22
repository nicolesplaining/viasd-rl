#!/bin/bash
# GRPO box: reuse policy_imitation.pt, run resumable GRPO (+ optional torch.compile) + bench.
# Auto-resumes from grpo_ckpt.pt. Re-run with larger ITERS to continue.
#   env: VIASD_VERIFIER(opt) ITERS G LAM N_TRAIN N_EVAL MAX_NEW COMPILE COMPILE_MODE OUT_DIR KEEP_MASK
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PYTHON=${PYTHON:-python3}
ITERS=${ITERS:-120}; G=${G:-6}; LAM=${LAM:-0.3}
N_TRAIN=${N_TRAIN:-100}; N_EVAL=${N_EVAL:-150}; MAX_NEW=${MAX_NEW:-320}
COMPILE=${COMPILE:-1}; COMPILE_MODE=${COMPILE_MODE:-reduce-overhead}
CARGS=""; [ "$COMPILE" = "1" ] && CARGS="--compile --compile_mode $COMPILE_MODE"
OUT_DIR=${OUT_DIR:-results/local}; mkdir -p "$OUT_DIR"
IMIT=${IMIT:-$OUT_DIR/policy_imitation.pt}; GRPO=${GRPO:-$OUT_DIR/policy_grpo.pt}
CKPT=${CKPT:-$OUT_DIR/grpo_ckpt.pt}; JSONL=${JSONL:-$OUT_DIR/grpo_log.jsonl}
KEEP_MASK=${KEEP_MASK:-}
KARG=""; [ -n "$KEEP_MASK" ] && KARG="--keep_mask $KEEP_MASK"

echo "=== GRPO box: iters=$ITERS G=$G lam=$LAM compile=$COMPILE/$COMPILE_MODE ==="; date
"$PYTHON" -m viasd.train_grpo --init "$IMIT" --out "$GRPO" \
    --ckpt "$CKPT" --jsonl "$JSONL" \
    --iters "$ITERS" --group_size "$G" --lam "$LAM" --r_correct 0.5 \
    --n_train "$N_TRAIN" --max_new "$MAX_NEW" --ckpt_every 10 --log_every 1 $KARG $CARGS

echo "=== BENCHMARK (via_rl column = GRPO policy) ==="
"$PYTHON" scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW" \
    --policy_imitation "$IMIT" --policy_rl "$GRPO" $KARG
echo "=== DONE ==="; date
