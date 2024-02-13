import logging
import random
from typing import Iterator

import torch
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LeiaDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int | None = None,
        disable_trans_token_loss: bool = False,
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._disable_trans_token_loss = disable_trans_token_loss

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self._tokenizer.pad(
            {"input_ids": [example["input_ids"] for example in examples]},
            padding="max_length",
            max_length=self._max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]

        labels = input_ids.clone()
        labels[labels == self._tokenizer.pad_token_id] = -100
        if "<trans>" in self._tokenizer.vocab:
            trans_start_token_id = self._tokenizer.vocab["<trans>"]
            trans_end_token_id = self._tokenizer.vocab["</trans>"]
            labels[labels == trans_start_token_id] = -100
            labels[labels == trans_end_token_id] = -100
            if self._disable_trans_token_loss:
                for row_index in range(input_ids.size(0)):
                    start_pos = None
                    for col_index in range(input_ids.size(1)):
                        input_id = input_ids[row_index, col_index]
                        if input_id == trans_start_token_id:
                            start_pos = col_index + 1
                        elif input_id == trans_end_token_id:
                            if start_pos is not None:
                                labels[row_index, start_pos:col_index] = -100
                            start_pos = None
                    if start_pos is not None:
                        labels[row_index, start_pos:] = -100

        batch["labels"] = labels

        return batch


class LeiaConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_size: int,
        max_length: int,
        max_num_examples: int,
        trans_start_token_id: int,
        trans_end_token_id: int,
        trans_insertion_prob: float,
        trans_insertion_strategy: str,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self._dataset = dataset
        self._dataset_size = dataset_size
        self._max_length = max_length
        self._max_num_examples = max_num_examples
        self._trans_start_token_id = trans_start_token_id
        self._trans_end_token_id = trans_end_token_id
        self._trans_insertion_prob = trans_insertion_prob
        self._trans_insertion_strategy = trans_insertion_strategy
        self._shuffle = shuffle
        self._seed = seed
        self._rnd = random.Random(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "LeiaConstantLengthDataset does not support multiprocessing"

        epoch_counter = 0
        dataset_example_counter = 0
        output_example_counter = 0
        token_counter = 0
        input_ids: list[int] = []
        trans_insertion_prob = self._trans_insertion_prob
        while True:
            dataset = self._dataset
            if self._shuffle:
                dataset = dataset.shuffle(self._seed + epoch_counter)
            dataset_iter = iter(dataset)

            while True:
                if len(input_ids) < self._max_length:
                    try:
                        example = next(dataset_iter)
                    except StopIteration:
                        break

                    dataset_example_counter += 1

                    input_ids += self._build_input_ids_from_example(example, trans_insertion_prob)

                    if dataset_example_counter != 1 and dataset_example_counter % 10000 == 0:
                        logger.info(
                            f"epoch: {epoch_counter} #token: {token_counter} progress: {(dataset_example_counter / self._dataset_size* 100):.2f}% insertion_prob: {trans_insertion_prob}"
                        )

                if len(input_ids) >= self._max_length:
                    yield {"input_ids": torch.tensor(input_ids[: self._max_length])}
                    token_counter += self._max_length
                    output_example_counter += 1
                    input_ids = input_ids[self._max_length :]

            if self._dataset_size is not None:
                logger.info(f"finished epoch {epoch_counter}")

            epoch_counter += 1

    def _build_input_ids_from_example(self, example: dict, trans_insertion_prob: float) -> list[int]:
        input_ids = example["input_ids"]
        if "entity_start_positions" in example:
            for start_position, end_position, entity_input_ids in zip(
                reversed(example["entity_start_positions"]),
                reversed(example["entity_end_positions"]),
                reversed(example["alternative_entity_input_ids"]),
            ):
                if self._rnd.random() > trans_insertion_prob:
                    continue
                entity_input_ids = [self._trans_start_token_id] + entity_input_ids + [self._trans_end_token_id]

                if self._trans_insertion_strategy == "left":
                    input_ids = input_ids[:start_position] + entity_input_ids + input_ids[start_position:]
                elif self._trans_insertion_strategy == "right":
                    input_ids = input_ids[:end_position] + entity_input_ids + input_ids[end_position:]
                elif self._trans_insertion_strategy == "replace":
                    input_ids = input_ids[:start_position] + entity_input_ids + input_ids[end_position:]
                else:
                    assert (
                        self._trans_insertion_strategy == "none"
                    ), f"Invalid strategy: {self._trans_insertion_strategy}"

        return input_ids
