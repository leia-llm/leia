#!/bin/bash

DIR_NAME_SUFFIX=$(echo ${MODEL_NAME} | cut -d "/" -f 2)

python scripts/build_wikipedia_dataset.py \
    --model_name ${MODEL_NAME} \
    --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/en_preprocessed" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv" \
    --output_dir "${WIKIPEDIA_DATA_DIR}/en_${DIR_NAME_SUFFIX}" \
    --max_length ${MAX_LENGTH} \
    --entity_vocab_size 1000000

for LANG in ${TARGET_LANGUAGES}
do
    if [ "${LANG}" != "en" ]; then
        python scripts/build_wikipedia_dataset.py \
            --model_name ${MODEL_NAME} \
            --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
            --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
            --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_${DIR_NAME_SUFFIX}" \
            --entity_vocab_file "${WIKIPEDIA_DATA_DIR}/en_${DIR_NAME_SUFFIX}/entity_vocab.tsv"
    fi
done
