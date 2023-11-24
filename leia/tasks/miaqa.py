import os
import string
from abc import ABCMeta, abstractmethod
from collections import Counter

from datasets import Dataset, load_dataset
from fugashi import Tagger

from .base import GenerationRequest, GenerationTask

_tagger = None


# The above functions are based on the following code:
# https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/9b6c999ba30c6db1cb8c5cc4f45c2a8027d78c0d/eval_scripts/eval_xor_full.py#L45
def normalize_answer(s: str):
    def remove_counter(text: str):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class MIAQABase(GenerationTask, metaclass=ABCMeta):
    LANGUAGE = ""

    def _get_train_dataset(self) -> None:
        return None

    @abstractmethod
    def _get_task_dataset(self) -> Dataset:
        pass

    def _example_to_text(self, example: dict) -> str:
        # Based on the prompt in the following code:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/93cbffa59180e74b5516927adef11b9eeb76bf28/lm_eval/tasks/nqopen.py#L62
        return f"Question: {example['question']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        answer = example["answers"][0]
        return " " + answer

    def _create_requests(self, example: dict, context: str) -> list[GenerationRequest]:
        max_generation_length = max(
            len(self._tokenizer.encode(answer, add_special_tokens=False)) for answer in example["answers"]
        )
        requests = [GenerationRequest(context, stop_sequences=["\n"], max_generation_length=max_generation_length)]
        return requests

    def _process_results(self, example: dict, results: list[str]) -> dict:
        generated_text = results[0]
        answers = example["answers"]

        if self.LANGUAGE == "ja":
            global _tagger
            if _tagger is None:
                _tagger = Tagger("-Owakati")
            # https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/9b6c999ba30c6db1cb8c5cc4f45c2a8027d78c0d/eval_scripts/eval_xor_full.py#L111C9-L115C80
            answers = [_tagger.parse(answer) for answer in answers]
            generated_text = _tagger.parse(generated_text.replace("・", " ").replace("、", ","))

        ret = {
            "exact_match": max(exact_match_score(generated_text, answer) for answer in answers),
            "f1": max(f1_score(generated_text, answer) for answer in answers),
            "prediction": generated_text,
        }
        return ret


def _create_mkqa_task_class(language: str) -> type[MIAQABase]:
    class _MKQA(MIAQABase, metaclass=ABCMeta):
        LANGUAGE = language

        def _get_task_dataset(self) -> Dataset:
            path = os.path.join(os.path.dirname(__file__), "data", f"mkqa-{language}.jsonl")
            dataset = load_dataset("json", data_files=path, split="train")
            dataset = dataset.filter(lambda example: example["answers"][0].lower() not in ("no answer", "yes", "no"))
            return dataset

    return _MKQA


def _create_xorqa_task_class(language: str) -> type[MIAQABase]:
    class _XORQA(MIAQABase):
        LANGUAGE = language

        def _get_task_dataset(self) -> Dataset:
            path = os.path.join(os.path.dirname(__file__), "data", "mia_2022_dev_xorqa.jsonl")
            dataset = load_dataset("json", data_files=path, split="train")
            dataset = dataset.filter(
                lambda example: example["lang"] == language
                and example["answers"][0].lower() not in ("no answer", "yes", "no")
            )
            return dataset

    return _XORQA


def get_task_mapping() -> dict[str, type[MIAQABase]]:
    tasks = {}
    for lang in ["ar", "en", "es", "fi", "ja", "km", "ko", "ms", "ru", "sv", "tr", "zh_cn"]:
        tasks[f"mkqa_{lang.replace('zh_cn', 'zh')}"] = _create_mkqa_task_class(lang)

    for lang in ["ar", "bn", "fi", "ja", "ko", "ru", "te"]:
        tasks[f"xorqa_{lang}"] = _create_xorqa_task_class(lang)

    return tasks
