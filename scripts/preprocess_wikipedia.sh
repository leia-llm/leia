#!/bin/bash

for LANG in ${TARGET_LANGUAGES}
do
    python scripts/preprocess_wikipedia.py \
        --wikiextractor_output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}" \
        --redirect_file "${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv" \
        --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
        --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed"
done
