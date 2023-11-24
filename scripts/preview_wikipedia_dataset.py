import argparse
import logging
import os

import datasets
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

BLUE = "\033[1;34m"
RESET = "\033[0;0m"


def print_example(example: dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase):
    cur = 0
    input_ids = example["input_ids"]
    entity_start_positions = example["entity_start_positions"]
    entity_end_positions = example["entity_end_positions"]
    alternative_entity_input_ids = example["alternative_entity_input_ids"]
    for start, end, alternative_input_ids in zip(
        entity_start_positions, entity_end_positions, alternative_entity_input_ids
    ):
        print(tokenizer.decode(input_ids[cur:start]), end="")
        mention_text = tokenizer.decode(input_ids[start:end])
        print(f"{BLUE} [{tokenizer.decode(alternative_input_ids)}] {mention_text} ", end="")
        cur = end
        print(RESET, end="")
    print(tokenizer.decode(input_ids[cur:]))


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = datasets.load_from_disk(args.dataset_dir)
    dataset = dataset.shuffle()
    for example in dataset:
        print_example(example, tokenizer)
        input("Press Enter to continue...")
        os.system("clear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
