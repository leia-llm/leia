import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class BelebeleBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return (
            f"Passage: {example['flores_passage']}\n"
            f"Question: {example['question']}\n"
            f"Answer:"
            # f"P: {example['flores_passage']}\n"
            # f"Q: {example['question']}\n"
            # f"A: {example['mc_answer1']}\n"
            # f"B: {example['mc_answer2']}\n"
            # f"C: {example['mc_answer3']}\n"
            # f"D: {example['mc_answer4']}\n"
            # f"Answer:"
        )

    def _example_to_target(self, example: dict) -> str:
        return " " + example[f"mc_answer{example['correct_answer_num']}"]
        # return " " + self._get_answer_letter(example)

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return int(example["correct_answer_num"]) - 1

    # @staticmethod
    # def _get_answer_letter(example: dict) -> str:
    #     return ["A", "B", "C", "D"][int(example["correct_answer_num"]) - 1]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + example[f"mc_answer{answer_num}"])
            for answer_num in range(1, 5)
            # LogLikelihoodRequest(context=context, continuation=" " + answer_letter)
            # for answer_letter in ["A", "B", "C", "D"]
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
            return load_dataset("facebook/belebele", language, split="test")

    return _Belebele


def get_task_mapping() -> dict[str, type[BelebeleBase]]:
    tasks = {}
    for lang, flores_lang in [
        ("en", "eng_Latn"),
        ("ja", "jpn_Jpan"),
        ("sw", "swh_Latn"),
        ("tr", "tur_Latn"),
        ("vi", "vie_Latn"),
        ("zh", "zho_Hans"),
    ]:
        tasks[f"belebele_{lang}"] = _create_task_class(flores_lang)

    return tasks
