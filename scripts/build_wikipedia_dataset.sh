#!/bin/bash

DIR_NAME_SUFFIX=$(echo ${MODEL_NAME} | cut -d "/" -f 2)

for LANG in ${TARGET_LANGUAGES}
do
    python scripts/build_wikipedia_dataset.py \
        --model_name ${MODEL_NAME} \
        --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
        --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv" \
        --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_dataset_${DIR_NAME_SUFFIX}" \
        ${ARGS}
done
