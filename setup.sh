#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ACR-GUI Environment Setup
# Based on the gui-aif conda environment (tmux session 0).
# Python 3.12 + PyTorch 2.5.1+cu124
# ============================================================

ENV_NAME="gui-aif"
PYTHON_VERSION="3.12"

# --- Step 1: Create conda env if it doesn't exist ---
if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[INFO] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# --- Step 2: Activate (for subsequent pip installs) ---
echo "[INFO] Activating ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# --- Step 3: Core ML stack ---
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.9.0
pip install vllm==0.6.6.post1
pip install accelerate==1.13.0
pip install deepspeed==0.15.4

# --- Step 4: Install gui-aif (open-r1) in editable mode ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src/gui-aif"
pip install -e ".[dev]"

# --- Step 5: Additional training / data dependencies ---
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install easyocr
pip install python-Levenshtein

# --- Step 6: Data download tools ---
pip install modelscope
pip install huggingface_hub
pip install hf_transfer

# --- Step 7: Optional: flash-attn (builds from source, needs CUDA dev tools) ---
# pip install flash-attn --no-build-isolation

# --- Step 8: Data processing libs ---
pip install datasets pyarrow Pillow

echo ""
echo "============================================================"
echo "[DONE] Environment '${ENV_NAME}' setup complete."
echo "Activate: conda activate ${ENV_NAME}"
echo "============================================================"
