# LEIA: Facilitating Cross-Lingual Knowledge Transfer in Language Models with Entity-based Data Augmentation

## Installation

```bash
pip install -r requirements.txt
pip install packaging
pip install flash-attn --no-build-isolation
pip install -e .
```

## Building Wikipedia Corpus Augmented with English Entity Names

The following commands build the Japanese Wikipedia corpus augmented with English entity names for LLaMA 2.
The dataset will be saved in `$WIKIPEDIA_DATASET_DIR`.

```bash
export LANG="ja"
export WIKIDATA_DATA_DIR="data/wikidata"
export WIKIPEDIA_DATA_DIR="data/wikipedia"
export WIKIDATA_DUMP_DATE="20230703"
export WIKIPEDIA_DUMP_DATE="20230701"

export MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
export WIKIPEDIA_DATASET_DIR="${WIKIPEDIA_DATA_DIR}/${LANG}_dataset"

mkdir -p ${WIKIDATA_DATA_DIR}
mkdir -p ${WIKIPEDIA_DATA_DIR}

wget "https://dumps.wikimedia.org/wikidatawiki/entities/${WIKIDATA_DUMP_DATE}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2" -P ${WIKIDATA_DATA_DIR}
wget https://dumps.wikimedia.org/enwiki/${WIKIPEDIA_DUMP_DATE}/enwiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 -P ${WIKIPEDIA_DATA_DIR}
wget https://dumps.wikimedia.org/${LANG}wiki/${WIKIPEDIA_DUMP_DATE}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 -P ${WIKIPEDIA_DATA_DIR}

wikiextractor \
    ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
    -o ${WIKIPEDIA_DATA_DIR}/${LANG} \
    --html-safe "" \
    --links \
    --no-templates

python scripts/extract_wikipedia_redirects.py \
    --dump_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
    --output_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv

python scripts/extract_wikidata_ids.py \
    --dump_file ${WIKIDATA_DATA_DIR}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2 \
    --output_dir ${WIKIDATA_DATA_DIR} \
    --languages "en,${LANG}"

python scripts/preprocess_wikipedia.py \
    --wikiextractor_output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}" \
    --redirect_file "${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
    --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed"


python scripts/build_wikipedia_dataset.py \
    --model_name ${MODEL_NAME_OR_PATH} \
    --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv" \
    --output_dir $WIKIPEDIA_DATASET_DIR
```

## Training

Fine-tuning with the dataset can be run using `train.sh`.
The trained checkpoint files will be available in `$OUTPUT_DIR`.

```bash
export RUN_NAME="leia_${LANG}"
export OUTPUT_DIR="runs/leia_${LANG}"

bash ./train.sh
```

### Evaluation

The experimental results on X-CODAH and X-CSQA can be obtained using `evaluate.sh`.

```bash
export MODEL_NAME_OR_PATH=${OUTPUT_DIR}
export TASK="xcodah_${LANG},xcsqa_${LANG}"
export NUM_FEWSHOT_SAMPLES="4,4"

bash ./evaluate.sh
```

Please refer to [llm-jp-eval](./llm-jp-eval/) and [JP Language Model Evaluation Harness](./jp-lm-evaluation-harness/) to reproduce the experimental results on Japanese datasets.
