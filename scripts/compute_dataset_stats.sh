#!/bin/bash

DIR_NAME_SUFFIX=$(echo ${MODEL_NAME} | cut -d "/" -f 2)

for LANG in ${TARGET_LANGUAGES}
do
    echo "LANGUAGE: ${LANG}"
    python scripts/compute_dataset_stats.py \
        --wikipedia_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_${DIR_NAME_SUFFIX}"
    echo "=========="
done
