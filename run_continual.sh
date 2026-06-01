#!/bin/bash
# -----------------------------------------------------------
# Continual GRPO Training Script
# Implements: Entropy-Weighted Spatial Consensus (R₁),
#            OCR Verification (R₂),
#            Visual-Semantic Similarity (R₃),
#            Domain-Aware KL Scheduling
# -----------------------------------------------------------
cd src/gui-aif

source activate gui-aif

export TASK_ALGO=continual_grpo
export TASK_TYPE=grounding
export TASK_MODEL=qwen25
export TASK_DATASET=mobile

export DEBUG_MODE="true"
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="continualGUI_${TASK_DATASET}"

# Dataset path
export DATA_PATH=/XYFS01/HDD_POOL/neu_chzhao/neu_chzhao_1/chuzhao/GUI-AiF-main/dataset.yaml

export OUTPUT_BASE_PATH=/XYFS01/HDD_POOL/neu_chzhao/neu_chzhao_1/chuzhao/GUI-AiF-main/ckpt

# Log
export LOG_DIR=${OUTPUT_BASE_PATH}/logs/${RUN_NAME}_${TIMESTAMP}/

# Save path
export SAVE_PATH=${OUTPUT_BASE_PATH}/saves/${RUN_NAME}/

export PYTHONPATH=src

# Model path
export CKPT_PATH=/XYFS01/HDD_POOL/neu_chzhao/neu_chzhao_1/chuzhao/models/qwen/Qwen2___5-VL-3B-Instruct

mkdir -p "${LOG_DIR}"
export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export N_NODE=1
export N_GPU_PER_NODE=4

echo "====== Continual GRPO Training ======"
echo "N_NODE: $N_NODE"
echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
echo "LOG_DIR: $LOG_DIR"
echo "DATA_PATH: $DATA_PATH"
echo "SAVE_PATH: $SAVE_PATH"

{
    echo "====== Continual GRPO Training ======"
    echo "N_NODE: $N_NODE"
    echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
    echo "LOG_DIR: $LOG_DIR"
    echo "DATA_PATH: $DATA_PATH"
    echo "SAVE_PATH: $SAVE_PATH"
} > "$LOG_PATH"

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29504}

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_COUNT \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# -----------------------------------------------------------
# Key hyperparameters aligned with the algorithm spec:
#   K            = 4    (samples per prompt; must match num_generations)
#   τ            = 0.7  (sampling temperature)
#   ε            = 0.2  (clipping parameter)
#   β₀           = 0.04 (initial KL penalty)
#   λ            = 5.0  (KL scheduling sensitivity)
#   N            = 50   (KL eval interval, steps)
#   n_eval       = 50   (eval samples per old domain)
#   m_min        = 50   (minimum crop side, pixels)
#   (α₁,α₂,α₃)  = (0.3, 0.5, 0.2)  — text instructions
#   (α₁',α₃')   = (0.4, 0.6)        — icon instructions
# -----------------------------------------------------------

torchrun $DISTRIBUTED_ARGS src/open_r1/continual_grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root . \
    --max_prompt_length 12048 \
    --num_generations 4 \
    --K 4 \
    --temperature 0.7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 100 \
    --bf16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing false \
    --attn_implementation eager \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --max_pixels 12845056 \
    --save_only_model false \
    --beta 0.04 \
    --learning_rate 1e-6 \
    --reward_funcs continual \
    --grid_H 10 \
    --grid_W 10 \
    --ocr_enabled true \
    --siglip_enabled true \
    --alpha1 0.3 \
    --alpha2 0.5 \
    --alpha3 0.2 \
    --alpha1_prime 0.4 \
    --alpha3_prime 0.6 \
    --kl_lambda 5.0 \
    --kl_N 50 \
    --kl_n_eval 50 \
    --m_min 50 \
    $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log"
