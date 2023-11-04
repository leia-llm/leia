#!/bin/bash

python scripts/extract_wikidata_ids.py \
    --dump_file ${WIKIDATA_DATA_DIR}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2 \
    --output_dir ${WIKIDATA_DATA_DIR} \
    --languages ${TARGET_LANGUAGES// /,}
