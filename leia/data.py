import logging
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
        entity_vocab: dict[str, int],
        max_length: int | None = None,
        max_entity_length: int | None = None,
        do_language_modeling: bool = False,
        padding: str = "max_length",
    ):
        self._tokenizer = tokenizer
        self._entity_vocab = entity_vocab
        self._max_length = max_length
        self._max_entity_length = max_entity_length
        self._do_language_modeling = do_language_modeling
        self._padding = padding

        assert padding in ("longest", "max_length"), "Padding must be either 'longest' or 'max_length'"
        if padding == "max_length":
            assert max_length is not None, "max_length must be specified if padding is 'max_length'"
            assert max_entity_length is not None, "max_entity_length must be specified if padding is 'max_length'"

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self._tokenizer.pad(
            {"input_ids": [example["input_ids"] for example in examples]},
            padding=self._padding,
            max_length=self._max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        max_entity_length = self._max_entity_length
        if self._padding == "longest":
            max_entity_length = max(len(example["entity_ids"]) for example in examples)

        batch["entity_ids"] = torch.full(
            (len(examples), max_entity_length), self._entity_vocab["<pad>"], dtype=torch.long
        )
        batch["entity_prev_token_positions"] = torch.zeros((len(examples), max_entity_length), dtype=torch.long)
        batch["entity_last_token_positions"] = torch.zeros((len(examples), max_entity_length), dtype=torch.long)

        for n, example in enumerate(examples):
            batch["entity_ids"][n, : example["entity_ids"].size(0)].copy_(example["entity_ids"])
            batch["entity_prev_token_positions"][n, : example["entity_prev_token_positions"].size(0)].copy_(
                example["entity_prev_token_positions"]
            )
            batch["entity_last_token_positions"][n, : example["entity_last_token_positions"].size(0)].copy_(
                example["entity_last_token_positions"]
            )

        if self._do_language_modeling:
            labels = batch["input_ids"].clone()
            labels[labels == self._tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch


class LeiaConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        max_length: int,
        max_entity_length: int,
        entity_vocab_size: int,
        infinite: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        dataset_size: int | None = None,
    ):
        self._dataset = dataset
        self._max_length = max_length
        self._max_entity_length = max_entity_length
        self._entity_vocab_size = entity_vocab_size
        self._infinite = infinite
        self._shuffle = shuffle
        self._seed = seed
        self._dataset_size = dataset_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "LeiaConstantLengthDataset does not support multiprocessing"

        buffer = []
        buffer_length = 0
        epoch_counter = 0
        example_counter = 0
        token_counter = 0
        current_example = None

        while True:
            dataset = self._dataset
            if self._shuffle:
                dataset = dataset.shuffle(self._seed + epoch_counter)
            dataset_iter = iter(dataset)

            while True:
                if current_example is None:
                    try:
                        current_example = next(dataset_iter)
                        example_counter += 1
                    except StopIteration:
                        break

                buffer_length += len(current_example["input_ids"])
                buffer.append(current_example)
                current_example = None
                if example_counter != 1 and example_counter % 10000 == 0 and self._dataset_size is not None:
                    logger.info(
                        f"epoch: {epoch_counter} #token: {token_counter} progress: {(example_counter/self._dataset_size * 100):.2f}%"
                    )

                if buffer_length >= self._max_length:
                    input_ids = []
                    entity_ids = []
                    entity_prev_token_positions = []
                    entity_last_token_positions = []
                    for example in buffer:
                        offset = len(input_ids)
                        input_ids += example["input_ids"]
                        if "entity_ids" in example and example["entity_ids"] is not None:
                            for entity_id, prev_token_position, last_token_position in zip(
                                example["entity_ids"],
                                example["entity_prev_token_positions"],
                                example["entity_last_token_positions"],
                            ):
                                if entity_id >= self._entity_vocab_size:
                                    continue
                                prev_token_position += offset
                                last_token_position += offset
                                if last_token_position < self._max_length:
                                    entity_ids.append(entity_id)
                                    entity_prev_token_positions.append(prev_token_position)
                                    entity_last_token_positions.append(last_token_position)

                    yield {
                        "input_ids": torch.tensor(input_ids[: self._max_length]),
                        "entity_ids": torch.tensor(entity_ids[: self._max_entity_length]),
                        "entity_prev_token_positions": torch.tensor(
                            entity_prev_token_positions[: self._max_entity_length]
                        ),
                        "entity_last_token_positions": torch.tensor(
                            entity_last_token_positions[: self._max_entity_length]
                        ),
                    }
                    token_counter += self._max_length

                    # The remaining part of the example is processed in the next batch
                    remaining_input_ids = input_ids[self._max_length :]
                    if remaining_input_ids:
                        current_example = {
                            "input_ids": remaining_input_ids,
                            "entity_ids": [],
                            "entity_prev_token_positions": [],
                            "entity_last_token_positions": [],
                        }
                        offset = len(buffer[-1]["input_ids"]) - len(remaining_input_ids)
                        for entity_id, prev_token_position, last_token_position in zip(
                            example["entity_ids"],
                            example["entity_prev_token_positions"],
                            example["entity_last_token_positions"],
                        ):
                            if prev_token_position >= offset:
                                prev_token_position -= offset
                                last_token_position -= offset
                                current_example["entity_ids"].append(entity_id)
                                current_example["entity_prev_token_positions"].append(prev_token_position)
                                current_example["entity_last_token_positions"].append(last_token_position)

                    buffer = []
                    buffer_length = 0

            if not self._infinite:
                break

            epoch_counter += 1
