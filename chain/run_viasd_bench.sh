#!/usr/bin/env bash
# run_viasd_bench.sh NEVAL MAXNEW CELLS  — Nicole's scripts/bench.py per GPU.
# CELL = gpu,policy_path,mask(none|file),verifier,tag
set -uo pipefail
cd ~/viasd
source ~/venv/bin/activate
export HF_HOME=~/.cache/huggingface
HF_TOKEN="$(cat ~/.hf_token 2>/dev/null || echo '')"; export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ~/bench_logs
NEVAL="${1:-30}"; MAXNEW="${2:-256}"; CELLS="$3"
IFS=';' read -ra A <<< "$CELLS"
for c in "${A[@]}"; do
  IFS=',' read -r gpu pol mask ver tag <<< "$c"
  mk=""; [ "$mask" != "none" ] && mk="--keep_mask $mask"
  CUDA_VISIBLE_DEVICES="$gpu" VIASD_VERIFIER="$ver" setsid nohup python scripts/bench.py \
    --n_eval "$NEVAL" --max_new "$MAXNEW" --policy_rl "$pol" --policy_imitation /nope $mk \
    > ~/bench_logs/$tag.log 2>&1 &
  echo "[launch] gpu=$gpu pol=$(basename $pol) mask=$mask -> $tag"
  sleep 2
done
echo "LAUNCHED ${#A[@]}"
