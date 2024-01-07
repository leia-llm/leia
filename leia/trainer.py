from typing import Any

import torch
from datasets import Dataset
from transformers.trainer import Trainer

from leia.tasks import get_task, LoglikelihoodTask, GenerationTask


class LeiaTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_tasks: list[str] | None = None,
        num_fewshot_samples_for_tasks: list[int] | None = None,
        eval_task_kwargs: dict[str, Any] | None = None,
        eval_log_likelihood_task_kwargs: dict[str, Any] | None = None,
        eval_generation_task_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._eval_tasks = []
        if eval_tasks is not None:
            self._eval_tasks = eval_tasks

        if num_fewshot_samples_for_tasks is None:
            self._num_fewshot_samples_for_tasks = [0] * len(self._eval_tasks)
        else:
            self._num_fewshot_samples_for_tasks = num_fewshot_samples_for_tasks
            assert len(num_fewshot_samples_for_tasks) == len(
                self._eval_tasks
            ), "The number of few-shot samples must match the number of tasks"

        self._eval_task_kwargs = {}
        if eval_task_kwargs is not None:
            self._eval_task_kwargs = eval_task_kwargs

        self._eval_log_likelihood_task_kwargs = {}
        if eval_log_likelihood_task_kwargs is not None:
            self._eval_log_likelihood_task_kwargs = eval_log_likelihood_task_kwargs

        self._eval_generation_task_kwargs = {}
        if eval_generation_task_kwargs is not None:
            self._eval_generation_task_kwargs = eval_generation_task_kwargs

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = {}
        if eval_dataset is not None or self.eval_dataset is not None:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self._eval_tasks:
            # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L1357
            model = self._wrap_model(self.model, training=False)
            model.eval()

            for task_name, num_fewshot_samples in zip(self._eval_tasks, self._num_fewshot_samples_for_tasks):
                task_class = get_task(task_name)
                task_kwargs = self._eval_task_kwargs
                if isinstance(task_class, LoglikelihoodTask):
                    task_kwargs.update(self._eval_log_likelihood_task_kwargs)
                elif isinstance(task_class, GenerationTask):
                    task_kwargs.update(self._eval_generation_task_kwargs)

                task_result = task_class(
                    model=model,
                    accelerator=self.accelerator,
                    tokenizer=self.tokenizer,
                    batch_size=self.args.per_device_eval_batch_size,
                    seed=self.args.seed,
                    num_fewshot_samples=num_fewshot_samples,
                    **task_kwargs,
                ).run()
                for metric_name, metric_value in task_result.metrics.items():
                    metric_key_name = f"{metric_key_prefix}_{task_name}_{metric_name}"
                    metrics[metric_key_name] = metric_value
                    self.log({metric_key_name: metric_value})

        return metrics
