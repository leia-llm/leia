import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class JNLI(LoglikelihoodTask):
    CHOICES: list[str] = ["entailment", "contradiction", "neutral"]
    DESCRIPTION: str = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JNLI", split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JNLI", split="validation")

    def _example_to_text(self, example: dict) -> str:
        choices = "\n".join(self.CHOICES)
        return (
            f"### 指示:\n与えられた前提と仮説の関係を回答してください。\n\n出力は以下から選択してください：\n{choices}\n\n"
            f"### 入力:\n前提：{example['sentence1']}\n仮説：{example['sentence2']}\n\n### 応答:\n"
        )

    def _example_to_target(self, example: dict) -> str:
        return self.CHOICES[example["label"]]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [LogLikelihoodRequest(context=context, continuation=choice) for choice in self.CHOICES]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}
