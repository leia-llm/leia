import random
from abc import ABCMeta

import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class XNLIBase(LoglikelihoodTask, metaclass=ABCMeta):
    LANGUAGE: str = ""
    QUESTION_WORD: str = ""
    ENTAILMENT_LABEL: str = ""
    NEUTRAL_LABEL: str = ""
    CONTRADICTION_LABEL: str = ""

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("xnli", self.LANGUAGE, split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("xnli", self.LANGUAGE, split="test")

    def _example_to_text(self, example: dict) -> str:
        return example["premise"] + ", " + self.QUESTION_WORD + "? [MASK], " + example["hypothesis"]

    def _example_to_target(self, example: dict) -> str:
        return " " + [self.ENTAILMENT_LABEL, self.NEUTRAL_LABEL, self.CONTRADICTION_LABEL][example["label"]]

    def _example_to_fewshot_prompt(self, example: dict) -> str:
        prompt = self._example_to_text(example)
        return prompt.replace("[MASK]", self._example_to_target(example).lstrip())

    def _create_context(
        self, example: dict, train_dataset: list[dict] | None, task_dataset: list[dict], rnd: random.Random
    ) -> str:
        context = self.DESCRIPTION

        if self._num_fewshot_samples != 0:
            fewshot_examples = rnd.sample(train_dataset, self._num_fewshot_samples)
            context += "\n\n".join([self._example_to_fewshot_prompt(fe) for fe in fewshot_examples])
            context += "\n\n"

        context += self._example_to_text(example)

        return context

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context="", continuation=context.replace("[MASK]", label))
            for label in [self.ENTAILMENT_LABEL, self.NEUTRAL_LABEL, self.CONTRADICTION_LABEL]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == example["label"]:
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


class XNLIAr(XNLIBase):  # Arabic
    LANGUAGE: str = "ar"
    QUESTION_WORD: str = "صحيح"
    ENTAILMENT_LABEL: str = "نعم"
    NEUTRAL_LABEL: str = "لذا"
    CONTRADICTION_LABEL: str = "رقم"


class XNLIBg(XNLIBase):  # Bulgarian
    LANGUAGE: str = "bg"
    QUESTION_WORD: str = "правилно"
    ENTAILMENT_LABEL: str = "да"
    NEUTRAL_LABEL: str = "така"
    CONTRADICTION_LABEL: str = "не"


class XNLIDe(XNLIBase):  # German
    LANGUAGE: str = "de"
    QUESTION_WORD: str = "richtig"
    ENTAILMENT_LABEL: str = "Ja"
    NEUTRAL_LABEL: str = "Auch"
    CONTRADICTION_LABEL: str = "Nein"


class XNLIEl(XNLIBase):  # Greek
    LANGUAGE: str = "el"
    QUESTION_WORD: str = "σωστός"
    ENTAILMENT_LABEL: str = "Ναί"
    NEUTRAL_LABEL: str = "Έτσι"
    CONTRADICTION_LABEL: str = "όχι"


class XNLIEn(XNLIBase):  # English
    LANGUAGE: str = "en"
    QUESTION_WORD: str = "right"
    ENTAILMENT_LABEL: str = "Yes"
    NEUTRAL_LABEL: str = "Also"
    CONTRADICTION_LABEL: str = "No"


class XNLIEs(XNLIBase):  # Spanish
    LANGUAGE: str = "es"
    QUESTION_WORD: str = "correcto"
    ENTAILMENT_LABEL: str = "Sí"
    NEUTRAL_LABEL: str = "Asi que"
    CONTRADICTION_LABEL: str = "No"


class XNLIFr(XNLIBase):  # French
    LANGUAGE: str = "fr"
    QUESTION_WORD: str = "correct"
    ENTAILMENT_LABEL: str = "Oui"
    NEUTRAL_LABEL: str = "Aussi"
    CONTRADICTION_LABEL: str = "Non"


class XNLIHi(XNLIBase):  # Hindi
    LANGUAGE: str = "hi"
    QUESTION_WORD: str = "सही"
    ENTAILMENT_LABEL: str = "हाँ"
    NEUTRAL_LABEL: str = "इसलिए"
    CONTRADICTION_LABEL: str = "नहीं"


class XNLIRu(XNLIBase):  # Russian
    LANGUAGE: str = "ru"
    QUESTION_WORD: str = "правильно"
    ENTAILMENT_LABEL: str = "Да"
    NEUTRAL_LABEL: str = "Так"
    CONTRADICTION_LABEL: str = "Нет"


class XNLISw(XNLIBase):  # Swahili
    LANGUAGE: str = "sw"
    QUESTION_WORD: str = "sahihi"
    ENTAILMENT_LABEL: str = "Ndiyo"
    NEUTRAL_LABEL: str = "Hivyo"
    CONTRADICTION_LABEL: str = "Hapana"


class XNLITh(XNLIBase):  # Thai
    LANGUAGE: str = "th"
    QUESTION_WORD: str = "ถูกต้อง"
    ENTAILMENT_LABEL: str = "ใช่"
    NEUTRAL_LABEL: str = "ดังนั้น"
    CONTRADICTION_LABEL: str = "ไม่"


class XNLITr(XNLIBase):  # Turkish
    LANGUAGE: str = "tr"
    QUESTION_WORD: str = "doğru"
    ENTAILMENT_LABEL: str = "Evet"
    NEUTRAL_LABEL: str = "Böylece"
    CONTRADICTION_LABEL: str = "Hayır"


class XNLIUr(XNLIBase):  # Urdu
    LANGUAGE: str = "ur"
    QUESTION_WORD: str = "صحیح"
    ENTAILMENT_LABEL: str = "جی ہاں"
    NEUTRAL_LABEL: str = "اس لئے"
    CONTRADICTION_LABEL: str = "نہیں"


class XNLIVi(XNLIBase):  # Vietnamese
    LANGUAGE: str = "vi"
    QUESTION_WORD: str = "đúng"
    ENTAILMENT_LABEL: str = "Vâng"
    NEUTRAL_LABEL: str = "Vì vậy"
    CONTRADICTION_LABEL: str = "Không"


class XNLIZh(XNLIBase):  # Chinese
    LANGUAGE: str = "zh"
    QUESTION_WORD: str = "正确"
    ENTAILMENT_LABEL: str = "是的"
    NEUTRAL_LABEL: str = "所以"
    CONTRADICTION_LABEL: str = "不是的"


def get_task_mapping() -> dict[str, type[XNLIBase]]:
    tasks = {}
    for name, value in globals().items():
        if name.startswith("XNLI") and name != "XNLIBase":
            task_name = f"{name[:-2]}_{name[-2:]}".lower()
            tasks[task_name] = value
    return tasks
