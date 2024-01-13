import random
from abc import ABCMeta

from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class PAWSXBase(LoglikelihoodTask, metaclass=ABCMeta):
    LANGUAGE: str = ""
    YES: str = ""
    NO: str = ""
    QUESTION_WORD: str = ""

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("paws-x", self.LANGUAGE, split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("paws-x", self.LANGUAGE, split="test")

    def _example_to_text(self, example: dict) -> str:
        return example["sentence1"] + ", " + self.QUESTION_WORD + "? [MASK], " + example["sentence2"]

    def _example_to_target(self, example: dict) -> str:
        return " " + [self.YES, self.NO][example["label"]]

    def _example_to_fewshot_prompt(self, example: dict) -> str:
        prompt = self._example_to_text(example)
        return prompt.replace("[MASK]", self._example_to_target(example).lstrip())

    def _create_context(
        self, example: dict, train_dataset: list[dict] | None, task_dataset: list[dict], rnd: random.Random
    ) -> str:
        context = self._get_description()

        if self._num_fewshot_samples != 0:
            fewshot_examples = rnd.sample(train_dataset, self._num_fewshot_samples)
            context += "\n\n".join([self._example_to_fewshot_prompt(fe) for fe in fewshot_examples])
            context += "\n\n"

        context += self._example_to_text(example)

        return context

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context="", continuation=context.replace("[MASK]", self.YES)),
            LogLikelihoodRequest(context="", continuation=context.replace("[MASK]", self.NO)),
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        ll_yes, ll_no = results
        prediction = int(ll_yes > ll_no)
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


class PAWSXDe(PAWSXBase):
    LANGUAGE: str = "de"
    YES: str = "Ja"
    NO: str = "Nein"
    QUESTION_WORD: str = "richtig"


class PAWSXEn(PAWSXBase):
    LANGUAGE: str = "en"
    YES: str = "Yes"
    NO: str = "No"
    QUESTION_WORD: str = "right"


class PAWSXEs(PAWSXBase):
    LANGUAGE: str = "es"
    YES: str = "Sí"
    NO: str = "No"
    QUESTION_WORD: str = "verdad"


class PAWSXFr(PAWSXBase):
    LANGUAGE: str = "fr"
    YES: str = "Oui"
    NO: str = "No"
    QUESTION_WORD: str = "right"


class PAWSXJa(PAWSXBase):
    LANGUAGE: str = "ja"
    YES: str = "はい"
    NO: str = "いいえ"
    QUESTION_WORD: str = "ですね"


class PAWSXKo(PAWSXBase):
    LANGUAGE: str = "ko"
    YES: str = "예"
    NO: str = "아니요"
    QUESTION_WORD: str = "맞죠"


class PAWSXZh(PAWSXBase):
    LANGUAGE: str = "zh"
    YES: str = "是"
    NO: str = "不是"
    QUESTION_WORD: str = "对吧"


def get_task_mapping() -> dict[str, type[PAWSXBase]]:
    tasks = {}
    for name, value in globals().items():
        if name.startswith("PAWSX") and name != "PAWSXBase":
            task_name = f"{name[:-2]}_{name[-2:]}".lower()
            tasks[task_name] = value
    return tasks
