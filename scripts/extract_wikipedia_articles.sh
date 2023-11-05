#!/bin/bash

for LANG in ${TARGET_LANGUAGES}
do
    wikiextractor \
        ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
        -o ${WIKIPEDIA_DATA_DIR}/${LANG} \
        --html-safe "" \
        --links \
        --no-templates \
        ${ARGS}
done
