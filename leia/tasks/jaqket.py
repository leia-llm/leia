import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class Jaqket(LoglikelihoodTask):
    def _get_train_dataset(self) -> Dataset:
        return load_dataset("kumapo/JAQKET", "v1.0", split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("kumapo/JAQKET", "v1.0", split="validation")

    def _example_to_text(self, example: dict) -> str:
        return f"Question: {example['question']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example["answer_candidates"][example["label"]]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + candidate)
            for candidate in example["answer_candidates"]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}
