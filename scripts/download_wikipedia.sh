#!/bin/bash

DATE="20230701"
TARGET_DIR="data/wikipedia"

cd ${TARGET_DIR}
for LANG in ar bg de el en es et eu fr hi ht id it ja ko my pt qu ru sw ta te th tr ur vi zh
do
    FILE_NAME="${LANG}wiki-${DATE}-pages-articles-multistream.xml.bz2"
    if [ ! -f ${FILE_NAME} ]
    then
        wget https://dumps.wikimedia.org/${LANG}wiki/${DATE}/${FILE_NAME}
    fi
done
