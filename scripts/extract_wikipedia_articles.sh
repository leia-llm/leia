#!/bin/bash

for LANG in ${TARGET_LANGUAGES}
do
    if [ ${LANG} = "zh" ]; then
        ARGS="--no-templates"
        echo "Using --no-templates"
    fi
    wikiextractor \
        ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
        -o ${WIKIPEDIA_DATA_DIR}/${LANG} \
        --html-safe "" \
        --links \
        ${ARGS}
done
