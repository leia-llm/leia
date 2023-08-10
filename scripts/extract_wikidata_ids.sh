#!/bin/bash

DATE="20230703"
TARGET_DIR="data/wikidata"

if [ -z "${TARGET_LANG}" ]; then
    TARGET_LANG="ar,bg,de,el,en,es,et,eu,fr,hi,ht,id,it,ja,ko,my,pt,qu,ru,sw,ta,te,th,tr,ur,vi,zh"
fi

python scripts/extract_wikidata_ids.py \
    --dump_file ${TARGET_DIR}/wikidata-${DATE}-all.json.bz2 \
    --output_dir ${TARGET_DIR} \
    --languages ${TARGET_LANG}
