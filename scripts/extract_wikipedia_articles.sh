#!/bin/bash

DATE="20230701"
TARGET_DIR="data/wikipedia"

if [ -z "${TARGET_LANG}" ]; then
    TARGET_LANG="ar en hi ja sw th tr vi zh"
fi

for LANG in ${TARGET_LANG}
do
    wikiextractor \
        ${TARGET_DIR}/${LANG}wiki-${DATE}-pages-articles-multistream.xml.bz2 \
        -o ${TARGET_DIR}/${LANG} \
        --html-safe "" \
        --links
done
