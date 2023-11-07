from abc import ABCMeta

import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class XStoryClozeBase(LoglikelihoodTask, metaclass=ABCMeta):
    def _example_to_text(self, example: dict) -> str:
        text = " ".join(
            [
                example["input_sentence_1"],
                example["input_sentence_2"],
                example["input_sentence_3"],
                example["input_sentence_4"],
            ]
        )
        return text

    def _example_to_target(self, example: dict) -> str:
        clozes = [example["sentence_quiz1"], example["sentence_quiz2"]]
        label = example["answer_right_ending"] - 1
        return " " + clozes[label]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context, continuation=" " + example["sentence_quiz1"]),
            LogLikelihoodRequest(context, continuation=" " + example["sentence_quiz2"]),
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        label = example["answer_right_ending"] - 1
        prediction = int(np.argmax(results))
        if prediction == label:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[XStoryClozeBase]:
    class _XStoryCloze(XStoryClozeBase):
        def _get_train_dataset(self) -> Dataset:
            return load_dataset("juletxara/xstory_cloze", language, split="train")

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("juletxara/xstory_cloze", language, split="eval")

    return _XStoryCloze


def get_task_mapping() -> dict[str, type[XStoryClozeBase]]:
    tasks = {}
    for lang in ["en", "ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]:
        tasks[f"xstory_cloze_{lang}"] = _create_task_class(lang)

    return tasks
