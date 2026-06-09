#!/bin/bash
cd src/gui_aif
export TASK_ALGO=sft
export TASK_TYPE=grounding
export TASK_MODEL=qwen25
export TASK_DATASET=mobile

export DEBUG_MODE="true"
export WANDB_MODE=offline


TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN_NAME="sft_baseline_${TASK_DATASET}"
export DATA_PATH=/GUI-AiF/dataset.yaml
export OUTPUT_BASE_PATH=/GUI-AiF/ckpt
export LOG_DIR=${OUTPUT_BASE_PATH}/logs/${RUN_NAME}_${TIMESTAMP}/
export SAVE_PATH=${OUTPUT_BASE_PATH}/saves/${RUN_NAME}/

# --- Base Model ---
export CKPT_PATH=/model/path/

export PYTHONPATH=src
mkdir -p "${LOG_DIR}"
export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="1,2,3,4"
export N_GPU_PER_NODE=4

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
MASTER_PORT=${MASTER_PORT:-29502} 
DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_COUNT \
    --master_port $MASTER_PORT
"

echo "Starting SFT Baseline Training..."
echo "SAVE_PATH: $SAVE_PATH"
echo "LOG_DIR: $LOG_DIR"

torchrun $DISTRIBUTED_ARGS src/open_r1/sft_baseline.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root . \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --logging_steps 100 \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --remove_unused_columns false \
    $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log"