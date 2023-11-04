#!/bin/bash

WIKIPEDIA_DATA_DIR="data/wikipedia"
WIKIDATA_DATA_DIR="data/wikidata"

if [ -z "${TARGET_LANG}" ]; then
  TARGET_LANG="ar en hi ja sw th tr vi zh"
fi

for LANG in ${TARGET_LANG}
do
  if [ -d "${WIKIPEDIA_DATA_DIR}/${LANG}_${OUTPUT_DIR_SUFFIX}" ]; then
    echo "Skipping ${LANG}"
  else
    python scripts/preprocess_wikipedia.py \
      --wikiextractor_output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}" \
      --redirect_file "${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv" \
      --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
      --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed"
  fi
done
