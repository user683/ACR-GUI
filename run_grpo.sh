cd src/gui-aif

export TASK_ALGO=grpo
export TASK_TYPE=grounding
export TASK_MODEL=qwen25
export TASK_DATASET=mobile


export DEBUG_MODE="true"
export WANDB_MODE=offline

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#RUN_NAME
RUN_NAME="continualGUI_${TASK_DATASET}"


# dataset path
export DATA_PATH=/GUI-AiF/dataset.yaml


export OUTPUT_BASE_PATH=/GUI-AiF/ckpt

# log
export LOG_DIR=${OUTPUT_BASE_PATH}/logs/${RUN_NAME}_${GPU_TYPE}_${TIMESTAMP}/

# save_path
export SAVE_PATH=${OUTPUT_BASE_PATH}/saves/${RUN_NAME}_${DATASET}_${GPU_TYPE}/

export PYTHONPATH=src


# Model path
export CKPT_PATH=


mkdir -p "${LOG_DIR}"

export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

# mkdir -p ${LOG_PATH}
# 1 16   2 16 4 8
export CUDA_VISIBLE_DEVICES="1,2,3,4"
export N_NODE=1
export N_GPU_PER_NODE=54


echo "N_NODE: $N_NODE"
echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
echo "LOG_DIR: $LOG_DIR"
echo "TASK_MEMO: $TASK_MEMO"
echo "DATA_PATH: $DATA_PATH"
echo "SAVE_PATH: $SAVE_PATH"

{
    echo "N_NODE: $N_NODE"
    echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
    echo "LOG_DIR: $LOG_DIR"
    echo "TASK_MEMO: $TASK_MEMO"
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
#GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
# GPU_COUNT=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_COUNT \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS src/open_r1/gaussian_grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root . \
    --max_prompt_length 12048 \
    --num_generations 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 100 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --max_pixels 12845056 \
    --save_only_model false \
    --beta 0.04  \
    --use_pairwise_diversity \
    --pairwise_diversity_weight 0.5 \
    --center_point_diversity_weight 15 \
    --use_center_point_diversity \
    --learning_rate 1e-6 $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log" 


    # --resume_from_checkpoint  \

    