import argparse
import math
import multiprocessing
import os
import shutil
from collections import Counter
from itertools import chain

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from leia.utils import load_tsv_mapping


def _encode_examples(
    text_list: list[str],
    anchors_list: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
    min_prev_token_position: int,
) -> dict[str, list[list[int | str]]]:
    encoded_texts = tokenizer(
        text_list,
        max_length=max_length,
        truncation=max_length is not None,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    ret = {
        "input_ids": encoded_texts["input_ids"],
        "wikidata_ids": [],
        "entity_prev_token_positions": [],
        "entity_last_token_positions": [],
        "length": [len(input_ids) for input_ids in encoded_texts["input_ids"]],
    }

    for text, anchors, offset_mapping, special_tokens_mask in zip(
        text_list,
        anchors_list,
        encoded_texts["offset_mapping"],
        encoded_texts["special_tokens_mask"],
    ):
        prev_char_start = -1
        char_token_mapping = {}
        for token_index, (char_start, char_end) in enumerate(offset_mapping):
            # same character offset can be set to multiple tokens
            if char_start != prev_char_start and special_tokens_mask[token_index] == 0:
                char_token_mapping[char_start] = token_index
                # a whitespace character is treated as part of the trailing token in sentencepiece
                if text[char_start] == " " and char_end - char_start > 1 and text[char_start + 1] != " ":
                    char_token_mapping[char_start + 1] = token_index
                prev_char_start = char_start
        char_token_mapping[offset_mapping[-1][1]] = len(offset_mapping)

        wikidata_ids = []
        entity_prev_token_positions = []
        entity_last_token_positions = []
        for anchor in anchors:
            wikidata_id = anchor["wikidata_id"]
            if wikidata_id is None:
                continue
            char_start = anchor["start"]
            char_end = anchor["end"]
            if char_start not in char_token_mapping or char_end not in char_token_mapping:
                continue

            prev_token_position = char_token_mapping[char_start] - 1
            if prev_token_position < min_prev_token_position:
                continue
            last_token_position = char_token_mapping[char_end] - 1

            wikidata_ids.append(wikidata_id)
            entity_prev_token_positions.append(prev_token_position)
            entity_last_token_positions.append(last_token_position)

        ret["wikidata_ids"].append(wikidata_ids)
        ret["entity_prev_token_positions"].append(entity_prev_token_positions)
        ret["entity_last_token_positions"].append(entity_last_token_positions)

    return ret


def _digitize_entities(
    wikidata_ids_list: list[list[str]],
    entity_prev_token_positions_list: list[list[int]],
    entity_last_token_positions_list: list[list[int]],
    entity_vocab: dict[str, int],
) -> dict[str, list[list[int]]]:
    ret: dict[str, list[list[int]]] = {
        "entity_ids": [],
        "entity_prev_token_positions": [],
        "entity_last_token_positions": [],
    }
    for wikidata_ids, entity_prev_token_positions, entity_last_token_positions in zip(
        wikidata_ids_list,
        entity_prev_token_positions_list,
        entity_last_token_positions_list,
    ):
        entity_ids = []
        new_entity_prev_token_positions = []
        new_entity_last_token_positions = []
        for wikidata_id, entity_prev_token_position, entity_last_token_position in zip(
            wikidata_ids,
            entity_prev_token_positions,
            entity_last_token_positions,
        ):
            if wikidata_id not in entity_vocab:
                continue
            entity_ids.append(entity_vocab[wikidata_id])
            new_entity_prev_token_positions.append(entity_prev_token_position)
            new_entity_last_token_positions.append(entity_last_token_position)

        ret["entity_ids"].append(entity_ids)
        ret["entity_prev_token_positions"].append(new_entity_prev_token_positions)
        ret["entity_last_token_positions"].append(new_entity_last_token_positions)

    return ret


def main(args: argparse.Namespace) -> None:
    if args.entity_vocab_file is None and args.entity_vocab_size is None:
        raise ValueError("Either entity_vocab_file or entity_vocab_size must be specified")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading dataset...")
    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    print("Preprocessing dataset...")
    dataset = dataset.map(
        _encode_examples,
        input_columns=["text", "anchors"],
        batched=True,
        remove_columns=["text", "title", "anchors", "language"],
        features=datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value(dtype="int16")),
                "wikidata_ids": datasets.Sequence(datasets.Value(dtype="string")),
                "entity_prev_token_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "entity_last_token_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "length": datasets.Value(dtype="int32"),
            }
        ),
        fn_kwargs={
            "tokenizer": tokenizer,
            "min_prev_token_position": args.min_prev_token_position,
            "max_length": args.max_length,
        },
        num_proc=args.num_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.entity_vocab_file is None:
        print("Building entity vocab...")
        entity_counter: Counter[str] = Counter()
        with tqdm(total=math.ceil(len(dataset) / 1000)) as pbar:
            for examples in dataset.select_columns("wikidata_ids").iter(batch_size=1000):
                entity_counter.update(chain.from_iterable(examples["wikidata_ids"]))
                pbar.update()

        entity_vocab_items = ["<pad>"]
        entity_vocab_items += [wikidata_id for wikidata_id, _ in entity_counter.most_common(args.entity_vocab_size - 1)]
        entity_vocab = {}
        with open(os.path.join(args.output_dir, f"entity_vocab.tsv"), "w") as f:
            for id_, wikidata_id in enumerate(entity_vocab_items):
                if wikidata_id == "<pad>":
                    count = 0
                else:
                    count = entity_counter[wikidata_id]
                f.write(f"{wikidata_id}\t{id_}\t{count}\n")
                entity_vocab[wikidata_id] = id_

        del entity_counter, entity_vocab_items

    else:
        print("Loading entity vocab...")
        entity_vocab = load_tsv_mapping(args.entity_vocab_file, int)
        shutil.copy(args.entity_vocab_file, os.path.join(args.output_dir, f"entity_vocab.tsv"))

    print("Finalizing dataset...")
    dataset = dataset.map(
        _digitize_entities,
        input_columns=["wikidata_ids", "entity_prev_token_positions", "entity_last_token_positions"],
        batched=True,
        remove_columns=["wikidata_ids"],
        features=datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value(dtype="int16")),
                "entity_ids": datasets.Sequence(datasets.Value(dtype="int32")),
                "entity_prev_token_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "entity_last_token_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "length": datasets.Value(dtype="int32"),
            }
        ),
        fn_kwargs={"entity_vocab": entity_vocab},
        num_proc=args.num_workers,
    )
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--preprocessed_dataset_dir", type=str, required=True)
    parser.add_argument("--wikidata_id_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--entity_vocab_file", type=str)
    parser.add_argument("--entity_vocab_size", type=int)
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--min_prev_token_position", type=int, default=5)
    args = parser.parse_args()

    main(args)
