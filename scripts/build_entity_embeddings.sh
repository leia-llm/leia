#!/bin/bash

accelerate launch \
  --mixed_precision fp16 \
  ${ACCELERATE_ARGS} \
  scripts/build_entity_embeddings.py \
  --wikipedia_dataset_dir ${WIKIPEDIA_DATASET_DIR} \
  --model_name ${MODEL_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --layers ${LAYERS} \
  ${ARGS}
