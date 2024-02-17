import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class XCODAHBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return ""

    def _example_to_target(self, example: dict) -> str:
        return example["question"]["choices"]["text"][self._get_answer_index(example)]

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return example["question"]["choices"]["label"].index(example["answerKey"])

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context="", continuation=text) for text in example["question"]["choices"]["text"]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == self._get_answer_index(example):
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[XCODAHBase]:
    class _XCODAH(XCODAHBase):
        def _get_train_dataset(self) -> None:
            return None

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("xcsr", f"X-CODAH-{language}", split="validation")

    return _XCODAH


def get_task_mapping() -> dict[str, type[XCODAHBase]]:
    tasks = {}
    for lang in ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "ur", "vi", "zh"]:
        tasks[f"xcodah_{lang}"] = _create_task_class(lang.replace("ja", "jap"))

    return tasks
