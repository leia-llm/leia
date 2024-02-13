#!/bin/bash

if [[ -z ${MODEL_NAME_OR_PATH} ]]; then
    echo "MODEL_NAME_OR_PATH is not set"
    exit 1
fi
if [[ -z ${TASK} ]]; then
    echo "TASK is not set"
    exit 1
fi
if [[ -z ${NUM_FEWSHOT_SAMPLES} ]]; then
    echo "NUM_FEWSHOT_SAMPLES is not set"
    exit 1
fi

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --dynamo_backend "no" \
    evaluate.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --task ${TASK} \
    --num_fewshot_samples ${NUM_FEWSHOT_SAMPLES} \
    --use_flash_attention_2
