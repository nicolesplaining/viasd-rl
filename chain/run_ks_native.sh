#!/usr/bin/env bash
set -uo pipefail
cd ~/work/knapspec
source ~/venv/bin/activate
export HF_HOME=~/.cache/huggingface
HF_TOKEN="$(cat ~/.hf_token 2>/dev/null || echo '')"; export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ~/bench_logs
M=Qwen/Qwen2.5-14B-Instruct
# GPU0: KnapSpec native, optimize-interval 64 (default); GPU7: interval 256 (Nicole's win)
for cell in "0,64,ks_int64" "7,256,ks_int256"; do
  IFS=',' read -r gpu interval tag <<< "$cell"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup python benchmark.py \
    --model "$M" --dataset gsm8k --max-length 512 --gamma 10 --skip-budget 12 \
    --optimize-interval "$interval" --num-samples 30 --compare-strategies \
    --out-dir "bench_out/$tag" > ~/bench_logs/$tag.log 2>&1 &
  echo "[launch] gpu=$gpu interval=$interval -> $tag"
  sleep 2
done
echo "KS_NATIVE_LAUNCHED"
