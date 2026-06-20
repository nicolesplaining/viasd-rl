#!/bin/bash
# GRPO box: reuse policy_imitation.pt, run resumable GRPO (+ optional torch.compile) + bench.
# Auto-resumes from grpo_ckpt.pt. Re-run with larger ITERS to continue.
#   env: VIASD_VERIFIER(opt) ITERS G LAM N_TRAIN N_EVAL MAX_NEW COMPILE COMPILE_MODE
set -e
cd ~/viasd-rl
source .venv/bin/activate
ITERS=${ITERS:-120}; G=${G:-6}; LAM=${LAM:-0.3}
N_TRAIN=${N_TRAIN:-100}; N_EVAL=${N_EVAL:-150}; MAX_NEW=${MAX_NEW:-320}
COMPILE=${COMPILE:-1}; COMPILE_MODE=${COMPILE_MODE:-reduce-overhead}
CARGS=""; [ "$COMPILE" = "1" ] && CARGS="--compile --compile_mode $COMPILE_MODE"

echo "=== GRPO box: iters=$ITERS G=$G lam=$LAM compile=$COMPILE/$COMPILE_MODE ==="; date
python -m viasd.train_grpo --init policy_imitation.pt --out policy_grpo.pt \
    --ckpt grpo_ckpt.pt --jsonl grpo_log.jsonl \
    --iters "$ITERS" --group_size "$G" --lam "$LAM" --r_correct 0.5 \
    --n_train "$N_TRAIN" --max_new "$MAX_NEW" --ckpt_every 10 --log_every 1 $CARGS

echo "=== BENCHMARK (via_rl column = GRPO policy) ==="
python scripts/bench.py --n_eval "$N_EVAL" --max_new "$MAX_NEW" --policy_rl policy_grpo.pt
echo "=== DONE ==="; date
