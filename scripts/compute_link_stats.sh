#!/bin/bash

for LANG in ${TARGET_LANGUAGES}
do
    echo "${LANG}:"
    python scripts/compute_link_stats.py \
        --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
        --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv"
done
