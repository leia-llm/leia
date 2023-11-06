#!/bin/bash

DIR_NAME_SUFFIX=$(echo ${MODEL_NAME} | cut -d "/" -f 2)

accelerate launch \
    --mixed_precision fp16 \
    --use_flash_attention_2 \
    scripts/build_entity_embeddings.py \
    --wikipedia_dataset_dir "${WIKIPEDIA_DATA_DIR}/en_${DIR_NAME_SUFFIX}" \
    --model_name ${MODEL_NAME} \
    --output_dir "${ENTITY_EMBEDDING_DIR}/en_${DIR_NAME_SUFFIX}" \
    --batch_size ${BATCH_SIZE:-"1"} \
    --layers ${LAYERS:-"16,24,28,31,32"} \
    ${ARGS}
