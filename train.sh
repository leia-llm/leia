#!/bin/bash

if [[ -z ${RUN_NAME} ]]; then
    echo "RUN_NAME is not set"
    exit 1
fi
if [[ -z ${MODEL_NAME_OR_PATH} ]]; then
    echo "MODEL_NAME_OR_PATH is not set"
    exit 1
fi
if [[ -z ${WIKIPEDIA_DATA_DIR} ]]; then
    echo "WIKIPEDIA_DATA_DIR is not set"
    exit 1
fi
if [[ -z ${LANGUAGE} ]]; then
    echo "LANGUAGE is not set"
    exit 1
fi

NUM_PROCESSES=${NUM_PROCESSES:-`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`}
BATCH_SIZE=${BATCH_SIZE:-"2048"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-"1"}    
GRADIENT_ACCUMULATION_STEPS=$((${BATCH_SIZE} / ${PER_DEVICE_TRAIN_BATCH_SIZE} / ${NUM_PROCESSES}))

MAX_LENGTH=${MAX_LENGTH:-"2048"}
if [[ -z ${MAX_STEPS} ]]; then
    MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-"204800000"}
    MAX_STEPS=$((${MAX_TRAIN_TOKENS} / ${BATCH_SIZE} / ${MAX_LENGTH}))
fi
EVAL_STEPS=${EVAL_STEPS:-$((${MAX_STEPS} / 5))}
SAVE_STEPS=${SAVE_STEPS:-$((${MAX_STEPS} / 5))}

DIR_NAME_SUFFIX=$(echo ${MODEL_NAME_OR_PATH} | cut -d "/" -f 2)

if [ `nvidia-smi | grep "V100" | wc -l` -eq 0 ]
then
    echo "Using BF16 & flash attention 2"
    ACCELERATE_ARGS="--mixed_precision bf16"
    ARGS="${ARGS} --bf16 true"
    if [[ -z ${NO_FLASH_ATTENTION_2} ]]; then
        ARGS="${ARGS} --use_flash_attention_2"
    fi
else
    ACCELERATE_ARGS="--mixed_precision fp16"
    ARGS="${ARGS} --fp16 true"
fi
 
if [[ ! -z ${EVAL_TASKS} ]]; then
    ARGS="${ARGS} --eval_tasks ${EVAL_TASKS}"
fi
if [[ ! -z ${NUM_FEWSHOT_SAMPLES_FOR_TASKS} ]]; then
    ARGS="${ARGS} --num_fewshot_samples_for_tasks ${NUM_FEWSHOT_SAMPLES_FOR_TASKS}"
fi
if [[ ! -z ${RESUME_FROM_CHECKPOINT} ]]; then
    ARGS="${ARGS} --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
fi
if [[ -z ${SEED} ]]; then
    SEED="42"
fi

accelerate launch \
    --num_machines 1 \
    --num_processes ${NUM_PROCESSES} \
    --dynamo_backend "no" \
    \
    --use_deepspeed \
    --zero_stage ${ZERO_STAGE:="3"} \
    --zero3_init_flag false \
    --zero3_save_16bit_model true \
    --offload_optimizer_device ${OFFLOAD_OPTIMIZER_DEVICE:-"none"} \
    \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --gradient_clipping ${GRADIENT_CLIPPING:-"1.0"} \
    \
    ${ACCELERATE_ARGS} \
    \
    train.py \
    \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR:-"runs/${RUN_NAME}"} \
    \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --wikipedia_dataset_dir ${WIKIPEDIA_DATASET_DIR:-"${WIKIPEDIA_DATA_DIR}/${LANGUAGE}_dataset_${DIR_NAME_SUFFIX}"} \
    \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE:-"2"} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-"true"} \
    --learning_rate ${LEARNING_RATE:-"2e-5"} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE:-"cosine"} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS:-"0"} \
    --weight_decay ${WEIGHT_DECAY:-"0.1"} \
    --adam_beta1 ${ADAM_BETA1:-"0.9"} \
    --adam_beta2 ${ADAM_BETA2:-"0.95"} \
    --adam_epsilon ${ADAM_EPSILON:-"1e-6"} \
    --eval_steps ${EVAL_STEPS} \
    --evaluation_strategy "steps" \
    \
    --max_length ${MAX_LENGTH} \
    \
    --trans_insertion_strategy ${TRANS_INSERTION_STRATEGY:-"none"} \
    --trans_insertion_prob  ${TRANS_INSERTION_PROB:-"1.0"} \
    --trans_insertion_prob_decay ${TRANS_INSERTION_PROB_DECAY:-"false"} \
    --trans_insertion_min_prob ${TRANS_INSERTION_MIN_PROB:-"0.0"} \
    --disable_trans_token_loss ${DISABLE_TRANS_TOKEN_LOSS:-"false"} \
    \
    --max_eval_samples_for_tasks ${MAX_EVAL_SAMPLES_FOR_TASKS:-"2000"} \
    --use_dynamic_generation_length ${USE_DYNAMIC_GENERATION_LENGTH:-"true"} \
    \
    --log_level "info" \
    --logging_steps "10" \
    \
    --save_strategy ${SAVE_STRATEGY:-"no"} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-"5"} \
    \
    --seed ${SEED} \
    --dataloader_num_workers "1" \
    --overwrite_output_dir "true" \
    --remove_unused_columns "false" \
    \
    $ARGS
