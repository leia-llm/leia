import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class BelebeleBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return f"Passage: {example['flores_passage']}\nQuestion: {example['question']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example[f"mc_answer{example['correct_answer_num']}"]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + example[f"mc_answer{answer_num}"])
            for answer_num in range(1, 5)
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == int(example["correct_answer_num"]) - 1:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(language: str) -> type[BelebeleBase]:
    class _Belebele(BelebeleBase):
        def _get_train_dataset(self) -> None:
            return None

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("facebook/belebele", language, split="train")

    return _Belebele


def get_task_mapping() -> dict[str, type[BelebeleBase]]:
    tasks = {}
    for lang, flores_lang in [
        ("ar", "arb_Arab"),
        ("en", "eng_Latn"),
        ("de", "deu_Latn"),
        ("es", "spa_Latn"),
        ("fr", "fra_Latn"),
        ("ja", "jpn_Jpan"),
        ("pt", "por_Latn"),
        ("ru", "rus_Cyrl"),
        ("sw", "swh_Latn"),
        ("tr", "tur_Latn"),
        ("vi", "vie_Latn"),
        ("zh", "zho_Hans"),
    ]:
        tasks[f"belebele_{lang}"] = _create_task_class(flores_lang)

    return tasks
