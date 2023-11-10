import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class ExamsBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return f"Question: {example['question']['stem']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example["question"]["choices"]["text"][self._get_answer_index(example)]

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return example["question"]["choices"]["label"].index(example["answerKey"])

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + choice)
            for choice in example["question"]["choices"]["text"]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == self._get_answer_index(example):
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[ExamsBase]:
    class _Exams(ExamsBase):
        def _get_train_dataset(self) -> Dataset:
            return load_dataset("exams", f"crosslingual_{language}", split="train")

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("exams", f"crosslingual_{language}", split="validation")

    return _Exams


def get_task_mapping() -> dict[str, type[ExamsBase]]:
    tasks = {}
    for lang in ["bg", "hr", "hu", "it", "mk", "pl", "pt", "sq", "sr", "tr", "vi"]:
        tasks[f"exams_{lang}"] = _create_task_class(lang)

    return tasks
