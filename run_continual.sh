#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------
# Continual GRPO Training Script
# Implements: Anchor-Collapse Regularization (ACR),
#            Domain-Aware KL Scheduling
# -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src/gui-aif"

export TASK_ALGO=continual_grpo
export TASK_TYPE=grounding
export TASK_MODEL=qwen25
export TASK_DATASET="${TASK_DATASET:-mobile}"

export DEBUG_MODE="${DEBUG_MODE:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME:-continualGUI_${TASK_DATASET}}"

# Dataset path
export DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/dataset_rico_train_500.yaml}"

export OUTPUT_BASE_PATH="${OUTPUT_BASE_PATH:-${SCRIPT_DIR}/ckpt}"

# Log
export LOG_DIR=${OUTPUT_BASE_PATH}/logs/${RUN_NAME}_${TIMESTAMP}/

# Save path
export SAVE_PATH="${SAVE_PATH:-${OUTPUT_BASE_PATH}/saves/${RUN_NAME}/}"

export PYTHONPATH=src

# Model path
export CKPT_PATH="${CKPT_PATH:-}"

if [ -z "${CKPT_PATH}" ]; then
    echo "ERROR: CKPT_PATH is empty. Set it to a Qwen2.5-VL checkpoint or model path."
    echo "Example: CKPT_PATH=/path/to/Qwen2.5-VL-3B-Instruct bash run_continual.sh"
    exit 2
fi

mkdir -p "${LOG_DIR}"
export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export N_NODE="${N_NODE:-1}"
export N_GPU_PER_NODE="${N_GPU_PER_NODE:-1}"

echo "====== Continual GRPO Training ======"
echo "N_NODE: $N_NODE"
echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
echo "LOG_DIR: $LOG_DIR"
echo "DATA_PATH: $DATA_PATH"
echo "SAVE_PATH: $SAVE_PATH"
echo "CKPT_PATH: $CKPT_PATH"

{
    echo "====== Continual GRPO Training ======"
    echo "N_NODE: $N_NODE"
    echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
    echo "LOG_DIR: $LOG_DIR"
    echo "DATA_PATH: $DATA_PATH"
    echo "SAVE_PATH: $SAVE_PATH"
    echo "CKPT_PATH: $CKPT_PATH"
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
#   K            = 16   (samples per prompt)
#   τ            = 0.7  (sampling temperature)
#   ε            = 0.2  (clipping parameter)
#   β₀           = 0.04 (initial KL penalty)
#   λ            = 5.0  (KL scheduling sensitivity)
#   N            = 50   (KL eval interval, steps)
#   n_eval       = 50   (eval samples per old domain)
#   λ_acr        = 0.2  (anchor regularization weight)
# -----------------------------------------------------------

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-local_scripts/zero3.json}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"
BETA="${BETA:-0.04}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ACR_ENABLED="${ACR_ENABLED:-true}"
MEMORY_JSONL="${MEMORY_JSONL:-}"
MEMORY_NPY="${MEMORY_NPY:-}"
MEMORY_WRITE_ENABLED="${MEMORY_WRITE_ENABLED:-true}"
MEMORY_WRITE_JSONL="${MEMORY_WRITE_JSONL:-}"
MEMORY_WRITE_NPY="${MEMORY_WRITE_NPY:-}"
LAMBDA_ACR="${LAMBDA_ACR:-0.2}"
# Step 2 (memory extraction) + Step 3 (capacity)
MEMORY_EXTRACT_MODE="${MEMORY_EXTRACT_MODE:-predict}"   # predict | gt
TAU_R="${TAU_R:-0.5}"
EXTRACT_MAX_NEW_TOKENS="${EXTRACT_MAX_NEW_TOKENS:-128}"
MEMORY_CAPACITY="${MEMORY_CAPACITY:-0}"                 # 0 = unbounded
MEMORY_PER_DOMAIN="${MEMORY_PER_DOMAIN:-true}"

torchrun $DISTRIBUTED_ARGS src/open_r1/continual_grpo.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root . \
    --max_prompt_length 12048 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --run_name $RUN_NAME \
    --save_steps "${SAVE_STEPS}" \
    --max_pixels "${MAX_PIXELS}" \
    --save_only_model false \
    --beta "${BETA}" \
    --learning_rate "${LEARNING_RATE}" \
    --reward_funcs ${REWARD_FUNCS:-gaussian_point gaussian_plane continual} \
    --acr_enabled "${ACR_ENABLED}" \
    --memory_jsonl "${MEMORY_JSONL}" \
    --memory_npy "${MEMORY_NPY}" \
    --memory_write_enabled "${MEMORY_WRITE_ENABLED}" \
    --memory_write_jsonl "${MEMORY_WRITE_JSONL}" \
    --memory_write_npy "${MEMORY_WRITE_NPY}" \
    --memory_extract_mode "${MEMORY_EXTRACT_MODE}" \
    --tau_r "${TAU_R}" \
    --extract_max_new_tokens "${EXTRACT_MAX_NEW_TOKENS}" \
    --memory_capacity "${MEMORY_CAPACITY}" \
    --memory_per_domain "${MEMORY_PER_DOMAIN}" \
    --lambda_acr "${LAMBDA_ACR}" \
    --kl_lambda 5.0 \
    --kl_N 50 \
    --kl_n_eval 50 \
    $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log"
