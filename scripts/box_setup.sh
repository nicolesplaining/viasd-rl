#!/usr/bin/env bash
# Phase 0 box provisioning: run on a fresh Lambda 1xH100 (80GB) instance.
# Order matters: CUDA sanity BEFORE any download/build (fail fast on driver mismatch).
set -euo pipefail

echo "=== 1. hardware sanity ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h / | tail -1   # need ~230GB free for 14B+32B HF + converted copies

echo "=== 2. venv + torch 2.5.1 cu121 ==="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE - STOP (driver mismatch?)"
x = torch.randn(512, 512, device="cuda"); y = x @ x
torch.cuda.synchronize()
print(f"OK torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

echo "=== 3. python deps ==="
pip install -q transformers==4.57.3 datasets safetensors sentencepiece 'huggingface_hub[hf_transfer]' accelerate

echo "=== 4. model downloads (tmux these; ~30GB total for phase 0-2) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p ~/ckpts
echo "run in tmux:"
echo "  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/ckpts/Qwen2.5-0.5B-Instruct"
echo "  huggingface-cli download Qwen/Qwen2.5-14B-Instruct  --local-dir ~/ckpts/Qwen2.5-14B-Instruct"
echo "  # 32B deferred behind gate G2:"
echo "  # huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir ~/ckpts/Qwen2.5-32B-Instruct"

echo "=== 5. after code rsync'd to ~/viasd-rl, convert + parity ladder ==="
echo "  python engine/convert_qwen.py --checkpoint_dir ~/ckpts/Qwen2.5-0.5B-Instruct"
echo "  python engine/parity.py --test features"
echo "  python engine/parity.py --test logits --hf-dir ~/ckpts/Qwen2.5-0.5B-Instruct --ckpt-dir ~/ckpts/Qwen2.5-0.5B-Instruct --model-name Qwen2.5-0.5B-Instruct"
echo "  python engine/convert_qwen.py --checkpoint_dir ~/ckpts/Qwen2.5-14B-Instruct"
echo "  python engine/parity.py --test logits --hf-dir ~/ckpts/Qwen2.5-14B-Instruct --ckpt-dir ~/ckpts/Qwen2.5-14B-Instruct --model-name Qwen2.5-14B-Instruct"
echo "  python engine/parity.py --test sd --drafter-dir ~/ckpts/Qwen2.5-0.5B-Instruct --verifier-dir ~/ckpts/Qwen2.5-14B-Instruct"
echo "BUDGET: note instance start time in ~/BUDGET.md; hard stop at \$250."
