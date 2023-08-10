import argparse
import logging
import os

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from leia.utils import load_tsv_mapping

logger = logging.getLogger(__name__)

BLUE = "\033[1;34m"
RESET = "\033[0;0m"


def print_example(example: dict, tokenizer: PreTrainedTokenizerBase, reverse_entity_vocab: dict[int, str]):
    cur = 0
    input_ids = example["input_ids"]
    entity_prev_token_positions = example["entity_prev_token_positions"]
    entity_last_token_positions = example["entity_last_token_positions"]
    entity_ids = example["entity_ids"]
    for start, end, entity_id in zip(entity_prev_token_positions, entity_last_token_positions, entity_ids):
        print(tokenizer.decode(input_ids[cur : start + 1]), end="")
        mention_text = tokenizer.decode(input_ids[start + 1 : end + 1])
        wikidata_id = reverse_entity_vocab.get(entity_id, "?")
        print(f"{BLUE} [{wikidata_id}] {mention_text} ", end="")
        cur = end + 1
        print(RESET, end="")
    print(tokenizer.decode(input_ids[cur:]))


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    entity_vocab_file = os.path.join(args.dataset_dir, "entity_vocab.tsv")
    entity_vocab = {}
    if os.path.exists(entity_vocab_file):
        entity_vocab = load_tsv_mapping(entity_vocab_file, int)
    reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}

    dataset = datasets.load_from_disk(args.dataset_dir)
    dataset = dataset.shuffle()
    for example in dataset:
        print_example(example, tokenizer, reverse_entity_vocab)
        input("Press Enter to continue...")
        os.system("clear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
