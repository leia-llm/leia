#!/bin/bash

wget "https://dumps.wikimedia.org/wikidatawiki/entities/${WIKIDATA_DUMP_DATE}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2" -P ${WIKIDATA_DATA_DIR}

for LANG in ${TARGET_LANGUAGES}
do
    wget https://dumps.wikimedia.org/${LANG}wiki/${WIKIPEDIA_DUMP_DATE}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 -P ${WIKIPEDIA_DATA_DIR}
done
