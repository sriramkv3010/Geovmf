#!/usr/bin/env bash
# scripts/install_torch.sh
# ========================
# Install a CUDA 11.8-compatible PyTorch build suitable for Python 3.12
# and NVIDIA P100/V100/A100 GPUs (e.g. Kaggle, Colab).
#
# Usage:
#   bash scripts/install_torch.sh
#
# For a CPU-only install, remove the +cu118 suffixes and the --index-url flag.

set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "==> Removing any pre-installed torch/torchvision/torchaudio ..."
"$PYTHON" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "==> Installing torch==2.3.1+cu118 ..."
"$PYTHON" -m pip install -q \
    torch==2.3.1+cu118 \
    torchvision==0.18.1+cu118 \
    torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "✅ PyTorch installed. Verify with:"
echo "   python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
