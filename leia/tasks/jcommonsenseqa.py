import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class JCommonsenseQA(LoglikelihoodTask):
    DESCRIPTION: str = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JCommonsenseQA", split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JCommonsenseQA", split="validation")

    def _example_to_text(self, example: dict) -> str:
        choices = "\n".join([f"- {example[f'choice{index}']}" for index in range(5)])
        return (
            f"### 指示:\n与えられた選択肢の中から、最適な答えを選んでください。出力は以下から選択してください：\n{choices}\n\n"
            f"### 入力:\n{example['question']}\n\n### 応答:\n"
        )

    def _example_to_target(self, example: dict) -> str:
        return example[f'choice{example["label"]}']

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [LogLikelihoodRequest(context=context, continuation=example[f"choice{index}"]) for index in range(5)]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}
