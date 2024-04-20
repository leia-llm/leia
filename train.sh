#!/bin/bash

if [[ -z ${RUN_NAME} ]]; then
    echo "RUN_NAME is not set"
    exit 1
fi
if [[ -z ${MODEL_NAME_OR_PATH} ]]; then
    echo "MODEL_NAME_OR_PATH is not set"
    exit 1
fi
if [[ -z ${WIKIPEDIA_DATASET_DIR} ]]; then
    echo "WIKIPEDIA_DATASET_DIR is not set"
    exit 1
fi

if [[ ! -z ${TASKS} ]]; then
    ARGS="${ARGS} --tasks ${TASKS}"
fi
if [[ ! -z ${NUM_FEWSHOT_SAMPLES} ]]; then
    ARGS="${ARGS} --num_fewshot_samples ${NUM_FEWSHOT_SAMPLES}"
fi

accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --dynamo_backend "no" \
    \
    --use_deepspeed \
    --zero_stage 3 \
    --zero3_init_flag false \
    --zero3_save_16bit_model true \
    --offload_optimizer_device ${OFFLOAD_OPTIMIZER_DEVICE:-"none"} \
    \
    --gradient_accumulation_steps 256 \
    --gradient_clipping 1.0 \
    --mixed_precision bf16 \
    \
    train.py \
    \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR:-"runs/${RUN_NAME}"} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --wikipedia_dataset_dir ${WIKIPEDIA_DATASET_DIR} \
    --entity_name_insertion_strategy ${ENTITY_NAME_INSERTION_STRATEGY:-"right"} \
    --entity_name_insertion_prob  ${ENTITY_NAME_INSERTION_PROB:-"0.5"} \
    --no_separator_tokens ${NO_SEPARATOR_TOKENS:-"false"} \
    \
    --per_device_train_batch_size "1" \
    --per_device_eval_batch_size "1" \
    --gradient_accumulation_steps "256" \
    --gradient_checkpointing "true" \
    --learning_rate ${LEARNING_RATE:-"5e-6"} \
    --lr_scheduler_type cosine \
    --max_steps "50" \
    --warmup_steps "0" \
    --weight_decay "0.1" \
    --adam_beta1 "0.9" \
    --adam_beta2 "0.95" \
    --adam_epsilon "1e-6" \
    --max_length "2048" \
    --log_level "info" \
    --logging_steps "10" \
    --seed ${SEED:-"42"} \
    --dataloader_num_workers "1" \
    --remove_unused_columns "false" \
    --bf16 true \
    --use_flash_attention_2 \
    \
    $ARGS
