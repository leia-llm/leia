from typing import Any

from datasets import Dataset
from transformers.trainer import Trainer

from leia.tasks import GenerationTask, LoglikelihoodTask, get_task


class LeiaTrainer(Trainer):
    def __init__(
        self,
        *args,
        tasks: list[str] | None = None,
        num_fewshot_samples: list[int] | None = None,
        task_kwargs: dict[str, Any] | None = None,
        log_likelihood_task_kwargs: dict[str, Any] | None = None,
        generation_task_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._tasks = []
        if tasks is not None:
            self._tasks = tasks

        if num_fewshot_samples is None:
            self._num_fewshot_samples = [0] * len(self._tasks)
        else:
            self._num_fewshot_samples = num_fewshot_samples
            assert len(num_fewshot_samples) == len(
                self._tasks
            ), "The number of few-shot samples must match the number of tasks"

        self._task_kwargs = {}
        if task_kwargs is not None:
            self._task_kwargs = task_kwargs

        self._log_likelihood_task_kwargs = {}
        if log_likelihood_task_kwargs is not None:
            self._log_likelihood_task_kwargs = log_likelihood_task_kwargs

        self._generation_task_kwargs = {}
        if generation_task_kwargs is not None:
            self._generation_task_kwargs = generation_task_kwargs

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = {}
        if eval_dataset is not None or self.eval_dataset is not None:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self._tasks:
            # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L1357
            model = self._wrap_model(self.model, training=False)
            model.eval()

            for task_name, num_fewshot_samples in zip(self._tasks, self._num_fewshot_samples):
                task_class = get_task(task_name)
                task_kwargs = self._task_kwargs
                if isinstance(task_class, LoglikelihoodTask):
                    task_kwargs.update(self._log_likelihood_task_kwargs)
                elif isinstance(task_class, GenerationTask):
                    task_kwargs.update(self._generation_task_kwargs)

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
