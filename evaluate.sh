#!/bin/bash

if [[ -z ${MODEL_NAME_OR_PATH} ]]; then
    echo "MODEL_NAME_OR_PATH is not set"
    exit 1
fi
if [[ -z ${TASKS} ]]; then
    echo "TASKS is not set"
    exit 1
fi
if [[ -z ${NUM_FEWSHOT_SAMPLES} ]]; then
    echo "NUM_FEWSHOT_SAMPLES is not set"
    exit 1
fi

if [[ ! -z ${OUTPUT_DIR} ]]; then
    ARGS="${ARGS} --output_dir ${OUTPUT_DIR}"
fi

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --dynamo_backend "no" \
    evaluate.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --tasks ${TASKS} \
    --num_fewshot_samples ${NUM_FEWSHOT_SAMPLES} \
    --use_flash_attention_2 \
    ${ARGS}
