#!/bin/bash

for LANG in ${TARGET_LANGUAGES}
do
    python scripts/extract_wikipedia_redirects.py \
        --dump_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
        --output_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv
done
