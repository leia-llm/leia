import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class HeadQABase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return f"Question: {example['qtext']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example["answers"][self._get_answer_index(example)]["atext"]

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return [answer["aid"] for answer in example["answers"]].index(example["ra"])

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + answer["atext"]) for answer in example["answers"]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == self._get_answer_index(example):
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[HeadQABase]:
    class _HeadQA(HeadQABase):
        def _get_train_dataset(self) -> Dataset:
            return load_dataset("head_qa", language, split="train")

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("head_qa", language, split="test")

    return _HeadQA


def get_task_mapping() -> dict[str, type[HeadQABase]]:
    tasks = {}
    for lang in ["en", "es"]:
        tasks[f"headqa_{lang}"] = _create_task_class(lang)

    return tasks
