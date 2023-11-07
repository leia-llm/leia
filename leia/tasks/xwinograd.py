from abc import ABCMeta

import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class XWinogradBase(LoglikelihoodTask, metaclass=ABCMeta):
    def _example_to_text(self, example: dict) -> str:
        return self._partial_context(example, example["option" + example["answer"]])

    def _example_to_target(self, example: dict) -> str:
        return self._partial_target(example)

    @staticmethod
    def _partial_context(example: dict, option: str) -> str:
        # Substitute the pronoun in the sentence with the specified option
        # and ignore everything after.
        pronoun_loc = example["sentence"].index("_")
        return example["sentence"][:pronoun_loc] + option

    @staticmethod
    def _partial_target(example: dict) -> str:
        # The target is everything after the document specified pronoun.
        pronoun_loc = example["sentence"].index("_") + 1
        return " " + example["sentence"][pronoun_loc:].strip()

    @staticmethod
    def _append_context(context: str, partial_context: str) -> str:
        contexts = context.split("\n\n")  # Each fewshot context is on its own new line.
        contexts.pop()  # Remove the correct context put in by `doc_to_text`.
        return "\n\n".join([*contexts, partial_context]) if contexts else partial_context

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        target = self._partial_target(example)
        requests = []
        for option in [example["option1"], example["option2"]]:
            partial_context = self._partial_context(example, option)
            full_context = self._append_context(context, partial_context)
            requests.append(LogLikelihoodRequest(full_context, continuation=target))

        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        if example["answer"] == "1":
            label = 0
        elif example["answer"] == "2":
            label = 1
        else:
            raise ValueError("Invalid answer label")

        prediction = int(np.argmax(results))
        if prediction == label:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[XWinogradBase]:
    class _XWinograd(XWinogradBase):
        def _get_train_dataset(self) -> None:
            return None

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("Muennighoff/xwinograd", language, split="test")

    return _XWinograd


def get_task_mapping() -> dict[str, type[XWinogradBase]]:
    tasks = {}
    for lang in ["en", "fr", "jp", "pt", "ru", "zh"]:
        tasks[f"xwinograd_{lang.replace('jp', 'ja')}"] = _create_task_class(lang)

    return tasks
