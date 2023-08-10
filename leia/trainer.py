from typing import Any

from datasets import Dataset
import torch
from torch import nn
from transformers.trainer import Trainer

from leia.tasks import get_task, LoglikelihoodTask, GenerationTask


class LeiaTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_tasks: list[str] | None = None,
        eval_task_kwargs: dict[str, Any] | None = None,
        eval_log_likelihood_task_kwargs: dict[str, Any] | None = None,
        eval_generation_task_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._eval_tasks = []
        if eval_tasks is not None:
            self._eval_tasks = eval_tasks

        self._eval_task_kwargs = {}
        if eval_task_kwargs is not None:
            self._eval_task_kwargs = eval_task_kwargs

        self._eval_generation_task_kwargs = {}
        if eval_generation_task_kwargs is not None:
            self._eval_generation_task_kwargs = eval_generation_task_kwargs

        self._lm_loss_count = torch.tensor(0, device=self.args.device)
        self._prev_token_loss_count = torch.tensor(0, device=self.args.device)
        self._last_token_loss_count = torch.tensor(0, device=self.args.device)

        self._lm_loss = torch.tensor(0.0, device=self.args.device)
        self._prev_token_loss = torch.tensor(0.0, device=self.args.device)
        self._last_token_loss = torch.tensor(0.0, device=self.args.device)

        self._prev_token_accuracy = torch.tensor(0.0, device=self.args.device)
        self._last_token_accuracy = torch.tensor(0.0, device=self.args.device)

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False) -> torch.Tensor | tuple:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if outputs.lm_loss is not None:
            self._lm_loss += outputs.lm_loss
            self._lm_loss_count += 1
        if outputs.entity_prev_token_loss is not None:
            self._prev_token_loss += outputs.entity_prev_token_loss
            self._prev_token_accuracy += outputs.entity_prev_token_accuracy
            self._prev_token_loss_count += 1
        if outputs.entity_last_token_loss is not None:
            self._last_token_loss += outputs.entity_last_token_loss
            self._last_token_accuracy += outputs.entity_last_token_accuracy
            self._last_token_loss_count += 1

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.control.should_log:
            log_dict = {}
            lm_loss_count = self.accelerator.gather(self._lm_loss_count).sum().item()
            if lm_loss_count > 0:
                lm_loss_scalar = self.accelerator.gather(self._lm_loss).sum().item()
                log_dict["lm_loss"] = round(lm_loss_scalar / lm_loss_count, 4)
                self._lm_loss = torch.tensor(0.0, device=self.args.device)
                self._lm_loss_count = torch.tensor(0, device=self.args.device)

            prev_token_loss_count = self.accelerator.gather(self._prev_token_loss_count).sum().item()
            if prev_token_loss_count > 0:
                prev_token_loss_scalar = self.accelerator.gather(self._prev_token_loss).sum().item()
                prev_token_accuracy_scalar = self.accelerator.gather(self._prev_token_accuracy).sum().item()
                log_dict["prev_token_loss"] = round(prev_token_loss_scalar / prev_token_loss_count, 4)
                log_dict["prev_token_accuracy"] = round(prev_token_accuracy_scalar / prev_token_loss_count, 4)
                self._prev_token_loss = torch.tensor(0.0, device=self.args.device)
                self._prev_token_accuracy = torch.tensor(0.0, device=self.args.device)
                self._prev_token_loss_count = torch.tensor(0, device=self.args.device)

            last_token_loss_count = self.accelerator.gather(self._last_token_loss_count).sum().item()
            if last_token_loss_count > 0:
                last_token_loss_scalar = self.accelerator.gather(self._last_token_loss).sum().item()
                last_token_accuracy_scalar = self.accelerator.gather(self._last_token_accuracy).sum().item()
                log_dict["last_token_loss"] = round(last_token_loss_scalar / last_token_loss_count, 4)
                log_dict["last_token_accuracy"] = round(last_token_accuracy_scalar / last_token_loss_count, 4)
                self._last_token_loss = torch.tensor(0.0, device=self.args.device)
                self._last_token_accuracy = torch.tensor(0.0, device=self.args.device)
                self._last_token_loss_count = torch.tensor(0, device=self.args.device)

            self.log(log_dict)
            self.control.should_log = True

        super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = {}
        if eval_dataset is not None:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L1357
        model = self._wrap_model(self.model, training=False)
        model.eval()

        for task_name in self._eval_tasks:
            task_class = get_task(task_name)
            task_kwargs = self._eval_task_kwargs
            if isinstance(task_class, GenerationTask):
                task_kwargs.update(self._eval_generation_task_kwargs)
            elif isinstance(task_class, LoglikelihoodTask):
                task_kwargs.update(self._eval_log_likelihood_task_kwargs)

            task_result = task_class(
                model=model,
                accelerator=self.accelerator,
                tokenizer=self.tokenizer,
                batch_size=self.args.per_device_eval_batch_size,
                seed=self.args.seed,
                **task_kwargs,
            ).run()
            for metric_name, metric_value in task_result.metrics.items():
                metric_key_name = f"{metric_key_prefix}_{task_name}_{metric_name}"
                metrics[metric_key_name] = metric_value
                self.log({metric_key_name: metric_value})

        return metrics
