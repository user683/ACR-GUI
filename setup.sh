# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the local open-r1 package used by the training scripts.
cd "$(dirname "${BASH_SOURCE[0]}")/src/gui-aif"
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install python-Levenshtein
# pip install flash-attn --no-build-isolation
