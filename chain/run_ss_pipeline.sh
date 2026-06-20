#!/usr/bin/env bash
# run_ss_pipeline.sh NTRAIN ITERS NEVAL MAXNEW QPMASK CELLS
# CELL = gpu,draftmask,lam,tag  — end-to-end on the KnapSpec self-spec draft (VIASD_DRAFT_MASK):
#   imitation -> per-token RL (ckpt every 10, jsonl per iter) -> bench. All reuse Nicole's code.
set -uo pipefail
NTRAIN="${1:-24}"; ITERS="${2:-40}"; NEVAL="${3:-15}"; MAXNEW="${4:-160}"; QP="${5}"; CELLS="${6}"
mkdir -p ~/bench_logs ~/ckpts
IFS=';' read -ra A <<< "$CELLS"
for c in "${A[@]}"; do
  IFS=',' read -r gpu draft lam tag <<< "$c"
  cat > /tmp/cell_$tag.sh <<CELL
cd ~/viasd; source ~/venv/bin/activate
export HF_HOME=~/.cache/huggingface HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null)
export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN USE_TF=0 USE_FLAX=0 TF_CPP_MIN_LOG_LEVEL=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIASD_DRAFTER=Qwen/Qwen2.5-0.5B-Instruct VIASD_VERIFIER=Qwen/Qwen2.5-14B-Instruct
export CUDA_VISIBLE_DEVICES=$gpu VIASD_DRAFT_MASK=$draft
L=~/bench_logs/ssp_$tag.log
echo "=== $tag draft=$draft qp=$QP lam=$lam :: IMITATION ===" > \$L
python -m viasd.train_imitation --n_train $NTRAIN --epochs 150 --max_new $MAXNEW --keep_mask $QP --out ~/ckpts/pi_$tag.pt >> \$L 2>&1
echo "=== $tag :: RL iters=$ITERS lam=$lam ckpt_every=10 ===" >> \$L
python -m viasd.train_rl_pertoken --init ~/ckpts/pi_$tag.pt --out ~/ckpts/ps_$tag.pt --ckpt ~/ckpts/ck_$tag.pt --jsonl ~/ckpts/jl_$tag.jsonl --iters $ITERS --batch 4 --lam $lam --r_correct 0.5 --max_new $MAXNEW --n_train $NTRAIN --keep_mask $QP --ckpt_every 10 >> \$L 2>&1
echo "=== $tag :: BENCH ===" >> \$L
python scripts/bench.py --n_eval $NEVAL --max_new $MAXNEW --keep_mask $QP --policy_rl ~/ckpts/ps_$tag.pt --policy_imitation /nope >> \$L 2>&1
echo "=== $tag :: DONE ===" >> \$L
CELL
  setsid nohup bash /tmp/cell_$tag.sh >/dev/null 2>&1 &
  echo "[launch] gpu=$gpu draft=$(basename $draft .json) lam=$lam -> ssp_$tag"
  sleep 2
done
echo "LAUNCHED ${#A[@]}"
