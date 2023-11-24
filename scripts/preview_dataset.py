import argparse
import logging
import os

import datasets
from transformers import AutoTokenizer

from leia.data import LeiaConstantLengthDataset

logger = logging.getLogger(__name__)

BLUE = "\033[1;34m"
RESET = "\033[0;0m"


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<trans>", "</trans>"]})

    dataset = datasets.load_from_disk(args.dataset_dir)
    dataset = dataset.shuffle()

    dataset = LeiaConstantLengthDataset(
        dataset=dataset,
        dataset_size=len(dataset),
        max_length=args.max_length,
        max_num_examples=len(dataset),
        trans_start_token_id=tokenizer.vocab["<trans>"],
        trans_end_token_id=tokenizer.vocab["</trans>"],
        trans_insertion_prob=args.trans_insertion_prob,
        trans_insertion_prob_decay=False,
        trans_insertion_strategy=args.trans_insertion_strategy,
    )
    for example in dataset:
        os.system("clear")
        text = tokenizer.decode(example["input_ids"])
        text = text.replace("<trans>", f"{BLUE}<trans>{RESET}")
        text = text.replace("</trans>", f"{BLUE}</trans>{RESET}")
        print(text)
        input("Press Enter to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--trans_insertion_prob", type=float, default=1.0)
    parser.add_argument(
        "--trans_insertion_strategy", type=str, choices=["random", "left", "right", "replace"], default="random"
    )
    args = parser.parse_args()

    main(args)
