from abc import ABCMeta

import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class XCopaBase(LoglikelihoodTask, metaclass=ABCMeta):
    LANGUAGE: str = ""
    CAUSE: str = "because"
    EFFECT: str = "therefore"

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("xcopa", self.LANGUAGE, split="validation")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("xcopa", self.LANGUAGE, split="test")

    def _example_to_text(self, example: dict) -> str:
        connector = {"cause": self.CAUSE, "effect": self.EFFECT}[example["question"]]
        return example["premise"].strip()[:-1] + f" {connector}"

    def _example_to_target(self, example: dict) -> str:
        if example["label"] == 0:
            correct_choice = example["choice1"]
        else:
            correct_choice = example["choice2"]
        return " " + self._convert_choice(correct_choice)

    @staticmethod
    def _convert_choice(choice: str) -> str:
        return choice[0].lower() + choice[1:]

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context, continuation=" " + self._convert_choice(example["choice1"])),
            LogLikelihoodRequest(context, continuation=" " + self._convert_choice(example["choice2"])),
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


class XCopaEt(XCopaBase):
    LANGUAGE: str = "et"
    CAUSE: str = "sest"
    EFFECT: str = "seetõttu"


class XCopaHt(XCopaBase):
    LANGUAGE: str = "ht"
    CAUSE: str = "poukisa"
    EFFECT: str = "donk sa"


class XCopaId(XCopaBase):
    LANGUAGE: str = "id"
    CAUSE: str = "karena"
    EFFECT: str = "maka"


class XCopaIt(XCopaBase):
    LANGUAGE: str = "it"
    CAUSE: str = "perché"
    EFFECT: str = "quindi"


class XCopaQu(XCopaBase):
    LANGUAGE: str = "qu"
    CAUSE: str = "imataq"
    EFFECT: str = "chaymi"


class XCopaSw(XCopaBase):
    LANGUAGE: str = "sw"
    CAUSE: str = "kwa sababu"
    EFFECT: str = "kwa hiyo"


class XCopaTa(XCopaBase):
    LANGUAGE: str = "ta"
    CAUSE: str = "காரணமாக"
    EFFECT: str = "எனவே"


class XCopaTh(XCopaBase):
    LANGUAGE: str = "th"
    CAUSE: str = "เพราะ"
    EFFECT: str = "ดังนั้น"


class XCopaTr(XCopaBase):
    LANGUAGE: str = "tr"
    CAUSE: str = "çünkü"
    EFFECT: str = "bu yüzden"


class XCopaVi(XCopaBase):
    LANGUAGE: str = "vi"
    CAUSE: str = "bởi vì"
    EFFECT: str = "vì vậy"


class XCopaZh(XCopaBase):
    LANGUAGE: str = "zh"
    CAUSE: str = "因为"
    EFFECT: str = "所以"


def get_task_mapping() -> dict[str, type[XCopaBase]]:
    tasks = {}
    for name, value in globals().items():
        if name.startswith("XCopa") and name != "XCopaBase":
            task_name = f"{name[:-2]}_{name[-2:]}".lower()
            tasks[task_name] = value
    return tasks
