import argparse
import multiprocessing
import re

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from leia.utils import load_tsv_mapping

entity_text_mapping = None


def _encode_examples(
    text_list: list[str],
    anchors_list: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None,
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
        "entity_start_positions": [],
        "entity_end_positions": [],
        "alternative_entity_input_ids": [],
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

        entity_start_positions = []
        entity_end_positions = []
        alternative_entity_input_ids = []
        for anchor in anchors:
            wikidata_id = anchor["wikidata_id"]
            if wikidata_id is None or wikidata_id not in entity_text_mapping:
                continue
            alternative_entity_text = entity_text_mapping[wikidata_id]
            if ":" in alternative_entity_text and alternative_entity_text.split(":")[0] in (
                "Wikipedia",
                "Category",
                "File",
                "Portal",
                "Template",
                "MediaWiki",
                "User",
                "Help",
                "Book",
                "Draft",
                "WikiProject",
                "Special",
                "Talk",
            ):
                continue
            if alternative_entity_text.startswith("List of"):
                continue
            alternative_entity_text = re.sub(r"\(.*?\)", "", alternative_entity_text).strip()
            entity_input_ids = tokenizer(alternative_entity_text, add_special_tokens=False)["input_ids"]
            alternative_entity_input_ids.append(entity_input_ids)

            char_start = anchor["start"]
            char_end = anchor["end"]
            if char_start not in char_token_mapping or char_end not in char_token_mapping:
                continue

            start_position = char_token_mapping[char_start]
            end_position = char_token_mapping[char_end]

            entity_start_positions.append(start_position)
            entity_end_positions.append(end_position)

        ret["alternative_entity_input_ids"].append(alternative_entity_input_ids)
        ret["entity_start_positions"].append(entity_start_positions)
        ret["entity_end_positions"].append(entity_end_positions)

    return ret


def main(args: argparse.Namespace) -> None:
    # entity_text_mapping is passed as a global variable for speed
    global entity_text_mapping

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)
    entity_text_mapping = {v: k for k, v in wikidata_id_mapping.items()}
    del wikidata_id_mapping

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
                "input_ids": datasets.Sequence(datasets.Value(dtype=args.input_ids_dtype)),
                "entity_start_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "entity_end_positions": datasets.Sequence(datasets.Value(dtype="int32")),
                "alternative_entity_input_ids": datasets.Sequence(
                    datasets.Sequence(datasets.Value(dtype=args.input_ids_dtype))
                ),
            }
        ),
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        },
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
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--input_ids_dtype", type=str, default="int16")
    args = parser.parse_args()

    main(args)
