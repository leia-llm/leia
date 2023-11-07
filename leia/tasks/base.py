import itertools
import random
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList


@dataclass
class Request:
    context: str


@dataclass
class LogLikelihoodRequest(Request):
    continuation: str


@dataclass
class GenerationRequest(Request):
    stop_sequences: list[str]
    max_generation_length: int


@dataclass
class TaskResult:
    metrics: dict[str, float]
    examples: list[dict]
    predictions: list[int] | list[str]


# Copied from https://github.com/Stability-AI/lm-evaluation-harness/blob/82ca7dd6f0eed2ea4ca957e73bb9c2048a2e5555/lm_eval/models/huggingface.py#L694
class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizerBase,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self._initial_decoder_input_length = initial_decoder_input_length
        self._done_tracker = [False] * batch_size
        self._sequence = sequence
        self._sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self._sequence_id_len = len(self._sequence_ids)
        self._tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, *args, **kwargs) -> bool:
        lookback_ids_batch = input_ids[:, self._initial_decoder_input_length :][:, -self._sequence_id_len :]
        lookback_tokens_batch = self._tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self._done_tracker):
            if not done:
                self._done_tracker[i] = self._sequence in lookback_tokens_batch[i]
        return False not in self._done_tracker


class BaseTask:
    DESCRIPTION: str = ""

    def __init__(
        self,
        model: PreTrainedModel,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_length: int,
        max_generation_length: int = 256,
        seed: int = 42,
        num_fewshot_samples: int = 0,
        max_samples: int | None = None,
        aggregation_function: Callable = np.mean,
    ):
        self._model = model
        self._accelerator = accelerator
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_generation_length = max_generation_length
        self._seed = seed
        self._num_fewshot_samples = num_fewshot_samples
        self._max_samples = max_samples
        self._aggregation_function = aggregation_function

    def run(self) -> TaskResult | None:
        with self._accelerator.main_process_first():
            train_dataset = self._get_train_dataset()
            if train_dataset is not None:
                train_dataset = list(train_dataset)
            task_dataset = list(self._get_task_dataset())

        rnd = random.Random(self._seed)
        rnd.shuffle(task_dataset)
        if self._max_samples is not None:
            task_dataset = task_dataset[: self._max_samples]

        all_requests = []
        for example in task_dataset:
            context = self._create_context(example, train_dataset=train_dataset, task_dataset=task_dataset, rnd=rnd)
            requests = self._create_requests(example, context)
            all_requests.append(requests)

        all_results = self._compute_results(list(itertools.chain(*all_requests)))

        all_metrics = []
        all_predictions = []
        cur = 0
        for example, requests in zip(task_dataset, all_requests):
            results = all_results[cur : cur + len(requests)]
            metrics = self._process_results(example, results)
            prediction = metrics.pop("prediction", None)

            all_metrics.append(metrics)
            all_predictions.append(prediction)
            cur += len(requests)

        final_metrics = self._aggregate_metrics(all_metrics)

        task_result = TaskResult(metrics=final_metrics, examples=task_dataset, predictions=all_predictions)
        return task_result

    @abstractmethod
    def _get_train_dataset(self) -> Dataset | None:
        pass

    @abstractmethod
    def _get_task_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def _example_to_text(self, example: dict) -> str:
        pass

    @abstractmethod
    def _example_to_target(self, example: dict) -> str:
        pass

    @abstractmethod
    def _create_requests(self, example: dict, context: str) -> list[Request]:
        pass

    @abstractmethod
    def _compute_results(self, requests: list[Request]) -> list:
        pass

    @abstractmethod
    def _process_results(self, example: dict, results: list[float | str]) -> dict:
        pass

    def _create_context(
        self, example: dict, train_dataset: list[dict] | None, task_dataset: list[dict], rnd: random.Random
    ) -> str:
        context = self.DESCRIPTION

        if self._num_fewshot_samples != 0:
            if train_dataset is not None:
                fewshot_examples = rnd.sample(train_dataset, self._num_fewshot_samples)
            else:
                fewshot_examples = rnd.sample(task_dataset, self._num_fewshot_samples + 1)
                fewshot_examples = [x for x in fewshot_examples if x != example][: self._num_fewshot_samples]

            context += "\n\n".join([self._example_to_text(fe) + self._example_to_target(fe) for fe in fewshot_examples])
            context += "\n\n"

        context += self._example_to_text(example)

        return context

    def _aggregate_metrics(self, metrics: list[dict[str, float]]) -> dict[str, float]:
        raw_metrics: dict[str, list[float]] = defaultdict(list)
        for metric in metrics:
            for key in metric.keys():
                raw_metrics[key].append(metric[key])

        ret = {key: self._aggregation_function(raw_metrics[key]) for key in raw_metrics.keys()}
        return ret

    def _is_deepspeed_zero_3(self) -> bool:
        # The following code is obtained from this URL:
        # https://github.com/huggingface/peft/blob/59778af504ddf368ae05cf9e009367cd872304e3/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L188
        if getattr(self._accelerator.state, "deepspeed_plugin", None) is not None:
            return self._accelerator.state.deepspeed_plugin.zero_stage == 3
        return False


class LoglikelihoodTask(BaseTask):
    def _compute_results(self, requests: list[LogLikelihoodRequest]) -> list[float]:
        inputs = []
        for request in requests:
            if request.context == "":
                context_ids = [self._tokenizer.eos_token_id]
            else:
                context_ids = self._tokenizer.encode(request.context, add_special_tokens=False)
            continuation_ids = self._tokenizer.encode(request.continuation, add_special_tokens=False)

            if len(context_ids) + len(continuation_ids) > self._max_length:
                context_ids = context_ids[-(self._max_length - len(continuation_ids) + 1) :]
            inputs.append((context_ids, continuation_ids))

        # longer inputs are processed first
        inputs_with_indices = sorted(enumerate(inputs), key=lambda x: len(x[1][0]) + len(x[1][1]), reverse=True)
        all_log_likelihoods = [None] * len(inputs_with_indices)
        pad_token_id = self._tokenizer.pad_token_id

        with self._accelerator.split_between_processes(
            inputs_with_indices, apply_padding=True
        ) as split_inputs_with_indices:
            for start_idx in trange(
                0, len(split_inputs_with_indices), self._batch_size, disable=not self._accelerator.is_local_main_process
            ):
                batch = split_inputs_with_indices[start_idx : start_idx + self._batch_size]
                input_ids = [
                    torch.tensor(
                        (context_ids + continuation_ids)[:-1], dtype=torch.long, device=self._accelerator.device
                    )
                    for _, (context_ids, continuation_ids) in batch
                ]
                input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=pad_token_id
                )
                input_ids_tensor = self._accelerator.pad_across_processes(
                    input_ids_tensor, dim=1, pad_index=pad_token_id
                )

                with torch.no_grad():
                    logits = self._model(input_ids_tensor).logits

                log_probs = F.log_softmax(logits, dim=-1)

                log_likelihoods = torch.empty(len(batch), device=self._accelerator.device)
                for index, (_, (context_ids, continuation_ids)) in enumerate(batch):
                    continuation_log_probs = log_probs[
                        index, len(context_ids) - 1 : len(context_ids) + len(continuation_ids) - 1
                    ]
                    log_likelihood = continuation_log_probs.gather(
                        1,
                        torch.tensor(continuation_ids, dtype=torch.long, device=self._accelerator.device).unsqueeze(1),
                    ).sum()
                    log_likelihoods[index] = log_likelihood

                log_likelihoods = self._accelerator.gather(log_likelihoods).cpu()

                indices = torch.tensor([x[0] for x in batch], device=self._accelerator.device)
                indices = self._accelerator.gather(indices).cpu()

                for idx, log_likelihood in zip(indices, log_likelihoods):
                    all_log_likelihoods[idx] = log_likelihood

        return all_log_likelihoods


class GenerationTask(BaseTask):
    def __init__(self, *args, use_dynamic_generation_length: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_dynamic_generation_length = use_dynamic_generation_length

    def _compute_results(self, requests: list[GenerationRequest]) -> list[str]:
        # longer inputs are processed first
        inputs_with_indices = sorted(
            enumerate(requests), key=lambda o: len(self._tokenizer.encode(o[1].context)), reverse=True
        )
        all_generated_texts = [None] * len(inputs_with_indices)
        pad_token_id = self._tokenizer.pad_token_id

        # we assume here that all stop sequences are the same in the batch
        stop_sequences = requests[0].stop_sequences + [self._tokenizer.eos_token]

        model = self._accelerator.unwrap_model(self._model)

        batch_size = 1
        with self._accelerator.split_between_processes(
            inputs_with_indices, apply_padding=True
        ) as split_inputs_with_indices:
            for start_idx in trange(
                0, len(split_inputs_with_indices), batch_size, disable=not self._accelerator.is_local_main_process
            ):
                batch = split_inputs_with_indices[start_idx : start_idx + self._batch_size]
                batch_requests: list[GenerationRequest] = [o[1] for o in batch]
                inputs = self._tokenizer(
                    [r.context for r in batch_requests], return_tensors="pt", padding=True, add_special_tokens=False
                )
                input_ids = inputs["input_ids"][:, self._max_generation_length - self._max_length :]
                input_ids = input_ids.to(self._accelerator.device)

                attention_mask = inputs["attention_mask"][:, self._max_generation_length - self._max_length :]
                attention_mask = attention_mask.to(self._accelerator.device)

                max_generation_length = self._max_generation_length
                if self._use_dynamic_generation_length and batch_requests[0].max_generation_length is not None:
                    max_generation_length = max(r.max_generation_length for r in batch_requests)

                stopping_criteria = StoppingCriteriaList(
                    [
                        MultiTokenEOSCriteria(sequence, self._tokenizer, input_ids.size(1), input_ids.size(0))
                        for sequence in stop_sequences
                    ]
                )
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_generation_length,
                        stopping_criteria=stopping_criteria,
                        do_sample=False,
                        synced_gpus=self._is_deepspeed_zero_3(),
                    )
                generated_ids = generated_ids[:, input_ids.size(1) :]
                generated_ids = self._accelerator.pad_across_processes(generated_ids, dim=1, pad_index=pad_token_id)
                generated_ids = self._accelerator.gather(generated_ids).cpu().tolist()

                indices = torch.tensor([x[0] for x in batch], device=self._accelerator.device)
                indices = self._accelerator.gather(indices).cpu()
                generated_texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for idx, response in zip(indices, generated_texts):
                    # remove stop sequence from the generated text
                    for stop_sequence in stop_sequences:
                        response = response.split(stop_sequence)[0]
                        all_generated_texts[idx] = response

        return all_generated_texts
