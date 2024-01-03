import json
import os
import random
import re
import unicodedata
from abc import ABCMeta

import numpy as np
from datasets import Dataset
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr

from .base import GenerationRequest, GenerationTask


class LLMJPEvalBase(GenerationTask, metaclass=ABCMeta):
    TASK_NAME = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_path = os.path.join(os.path.dirname(__file__), "data", "llm-jp-eval", f"{self.TASK_NAME}.json")
        with open(data_path) as f:
            self._data = json.load(f)

    def _get_train_dataset(self) -> Dataset:
        return Dataset.from_list(self._data["few_shots"])

    def _get_task_dataset(self) -> Dataset:
        return Dataset.from_list(self._data["samples"])

    def _example_to_text(self, example: dict) -> str:
        return f"### 入力：\n{example['input']}\n### 回答：\n"

    def _example_to_target(self, example: dict) -> str:
        return example["output"]

    def _create_context(
        self, example: dict, train_dataset: list[dict], task_dataset: list[dict], rnd: random.Random
    ) -> str:
        context = f"以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。\n### 指示\n{self._data['instruction']}\n\n"

        if self._num_fewshot_samples != 0:
            fewshot_examples = train_dataset[: self._num_fewshot_samples]
            context += "\n\n".join([self._example_to_text(fe) + self._example_to_target(fe) for fe in fewshot_examples])
            context += "\n\n"

        context += self._example_to_text(example)
        return context

    def _create_requests(self, example: dict, context: str) -> list[GenerationRequest]:
        max_generation_length = self._data["output_length"]
        requests = [GenerationRequest(context, stop_sequences=["\n"], max_generation_length=max_generation_length)]
        return requests

    def _process_results(self, example: dict, results: list[str]) -> dict:
        predicted = self._normalize(results[0].split("\n")[0])
        gold = self._normalize(example["output"])
        ret = {"prediction": predicted}

        for metric in self._data["metrics"]:
            if metric == "exact_match":
                ret[metric] = predicted == gold
            elif metric == "char_f1":
                ret[metric] = fuzz.token_sort_ratio(predicted, gold) / 100.0
            elif metric in ("spearman", "pearson"):
                ret[metric] = (self._parse_float(predicted), float(gold))
            else:
                raise RuntimeError(f"Unknown metric: {metric}")

        return ret

    def _aggregate_metrics(self, metrics: list[dict]) -> dict[str, float]:
        ret = {}
        for metric_name in self._data["metrics"]:
            if metric_name in {"spearman", "pearson"}:
                predicted_scores = [m[metric_name][0] for m in metrics]
                gold_scores = [m[metric_name][1] for m in metrics]
                if metric_name == "pearson":
                    ret[metric_name] = pearsonr(gold_scores, predicted_scores)[0]
                else:
                    ret[metric_name] = spearmanr(gold_scores, predicted_scores)[0]
            else:
                ret[metric_name] = np.mean([m[metric_name] for m in metrics])

        return ret

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    # Copied from https://github.com/llm-jp/llm-jp-eval/blob/v1.0.0/src/llm_jp_eval/utils.py#L8
    @staticmethod
    def _parse_float(text: str) -> float:
        cleaned_str = re.sub(r"[^0-9.]", "", text)
        try:
            return float(cleaned_str)
        except ValueError:
            return -2.0


def _create_task_class(task_name: str) -> type[LLMJPEvalBase]:
    class _LLMJPEvalTask(LLMJPEvalBase):
        TASK_NAME = task_name

    return _LLMJPEvalTask


def get_task_mapping() -> dict[str, type[LLMJPEvalBase]]:
    tasks = {}
    for task_name in (
        "jamp",
        "janli",
        # "jcommonsenseqa",
        "jemhopqa",
        "jnli",
        "jsem",
        "jsick",
        # "jsquad",
        "jsts",
        "niilc",
    ):
        tasks[task_name] = _create_task_class(task_name)

    return tasks
