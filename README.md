<div align="center">
  <img src="resources/logo.png" style="width: 60%" alt="LEIA">
</div>

This repository hosts the source code for our paper, [**LEIA: Facilitating Cross-Lingual Knowledge Transfer in Language Models with Entity-based Data Augmentation**](https://arxiv.org/abs/2402.11485). This paper proposes a new language adaptation technique for large language models (LLMs) aimed at enhancing _cross-lingual knowledge transfer_ from English to other languages.

By augmenting the Wikipedia-based training corpus in the target language with English entity names placed alongside their corresponding entity hyperlinks, LEIA enables an LLM to extract and apply its internal English knowledge about the entities within the target language text during training. The following figure illustrates the proposed data augmentation of LEIA applied to text from Chinese Wikipedia. English entity names enclosed within special `<translate>` and `</translate>` tokens are placed next to hyperlinks to facilitate cross-lingual transfer from English to Chinese.

<div align="center">
  <img src="resources/overview.svg" alt="Data augmentation example of LEIA">
</div>

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
# Target language
export LANG="ja"

# Model to be fine-tuned
export MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"

# Directories for storing Wikidata and Wikipedia data
export WIKIDATA_DATA_DIR="data/wikidata"
export WIKIPEDIA_DATA_DIR="data/wikipedia"

# Dump dates for Wikidata and Wikipedia
export WIKIDATA_DUMP_DATE="20230703"
export WIKIPEDIA_DUMP_DATE="20230701"

# Directory for storing the training dataset
export WIKIPEDIA_DATASET_DIR="${WIKIPEDIA_DATA_DIR}/${LANG}_dataset"

# Create directories for Wikidata and Wikipedia data
mkdir -p ${WIKIDATA_DATA_DIR}
mkdir -p ${WIKIPEDIA_DATA_DIR}

# Download Wikidata and Wikipedia dumps
wget https://dumps.wikimedia.org/wikidatawiki/entities/${WIKIDATA_DUMP_DATE}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2 -P ${WIKIDATA_DATA_DIR}
wget https://dumps.wikimedia.org/${LANG}wiki/${WIKIPEDIA_DUMP_DATE}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 -P ${WIKIPEDIA_DATA_DIR}

# Process Wikipedia dump using WikiExtractor
wikiextractor \
    ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
    -o ${WIKIPEDIA_DATA_DIR}/${LANG} \
    --html-safe "" \
    --links \
    --no-templates

# Extract Wikipedia redirect information
python scripts/extract_wikipedia_redirects.py \
    --dump_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-${WIKIPEDIA_DUMP_DATE}-pages-articles-multistream.xml.bz2 \
    --output_file ${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv

# Extract inter-language link data from Wikidata dump
python scripts/extract_wikidata_ids.py \
    --dump_file ${WIKIDATA_DATA_DIR}/wikidata-${WIKIDATA_DUMP_DATE}-all.json.bz2 \
    --output_dir ${WIKIDATA_DATA_DIR} \
    --languages "en,${LANG}"

# Preprocess Wikipedia corpus
python scripts/preprocess_wikipedia.py \
    --wikiextractor_output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}" \
    --redirect_file "${WIKIPEDIA_DATA_DIR}/${LANG}wiki-redirects.tsv" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/${LANG}-wikidata-ids.tsv" \
    --output_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed"

# Build Wikipedia dataset for training
python scripts/build_wikipedia_dataset.py \
    --model_name ${MODEL_NAME_OR_PATH} \
    --preprocessed_dataset_dir "${WIKIPEDIA_DATA_DIR}/${LANG}_preprocessed" \
    --wikidata_id_file "${WIKIDATA_DATA_DIR}/en-wikidata-ids.tsv" \
    --output_dir ${WIKIPEDIA_DATASET_DIR}
```

The training dataset can be seen with the following command:

```bash
python scripts/preview_dataset.py \
    --model_name ${MODEL_NAME_OR_PATH} \
    --dataset_dir ${WIKIPEDIA_DATASET_DIR}
```

## Training

To fine-tune the model with the previously built dataset, run the `train.sh` script.
This will save the trained model checkpoints in the specified output directory.

```bash
# Name for this training run
export RUN_NAME="leia_${LANG}"
# Output directory for saving the model checkpoint files
export OUTPUT_DIR="runs/leia_${LANG}"

# Start training
./train.sh
```

## Evaluation

To obtain experimental results, run the `evaluate.sh` script.

```bash
# Model path to be evaluated
export MODEL_NAME_OR_PATH=${OUTPUT_DIR}
# Tasks to be evaluated
export TASKS="xcodah_${LANG},xcsqa_${LANG}"
# Number of fewshot samples for each task
export NUM_FEWSHOT_SAMPLES="0,4"

# Run evaluation
./evaluate.sh
```

For replicating the experimental results on Japanese datasets, refer to [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval/tree/v1.0.0) and [JP Language Model Evaluation Harness](https://github.com/Stability-AI/lm-evaluation-harness/tree/9b42d412d040084bb270ca87199b7da9a91a9d7d).

## Citation

```bibtex
@article{Yamada2024LEIA,
  title={LEIA: Facilitating Cross-Lingual Knowledge Transfer in Language Models with Entity-based Data Augmentation},
  author={Ikuya Yamada and Ryokan Ri},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.11485}
}
```
