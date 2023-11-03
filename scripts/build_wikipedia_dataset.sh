#!/bin/bash

WIKIPEDIA_DATA_DIR="data/wikipedia"
WIKIDATA_DATA_DIR="data/wikidata"

if [ -z "${MODEL_NAME}" ]; then
  MODEL_NAME="meta-llama/Llama-2-7b-hf"
fi
if [ -z "${MAX_LENGTH}" ]; then
  MAX_LENGTH=4096
fi
if [ -z "${TARGET_LANG}" ]; then
  TARGET_LANG="ar hi ja sw th tr vi zh"
fi

OUTPUT_DIR_SUFFIX=$(echo ${MODEL_NAME} | cut -d "/" -f 2)

if [ -d "${WIKIPEDIA_DATA_DIR}/en_${OUTPUT_DIR_SUFFIX}" ]; then
  echo "Skipping en"
else
  python scripts/build_wikipedia_dataset.py \
    --model_name ${MODEL_NAME} \
    --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/en_preprocessed" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv" \
    --output_dir "${WIKIPEDIA_DATA_DIR}/en_${OUTPUT_DIR_SUFFIX}" \
    --max_length ${MAX_LENGTH} \
    --entity_vocab_size 1000000
fi

for LANG in ${TARGET_LANG}
do
  if [ -d "${WIKIPEDIA_DATA_DIR}/${LANG}_${OUTPUT_DIR_SUFFIX}" ]; then
    echo "Skipping ${LANG}"
  else
    python scripts/build_wikipedia_dataset.py \
      --model_name ${MODEL_NAME} \
      --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
      --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
      --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_${OUTPUT_DIR_SUFFIX}" \
      --entity_vocab_file "${WIKIPEDIA_DATA_DIR}/en_${OUTPUT_DIR_SUFFIX}/entity_vocab.tsv"
  fi
done
