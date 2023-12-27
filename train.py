import logging
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from leia.data import LeiaConstantLengthDataset, LeiaDataCollator
from leia.trainer import LeiaTrainer

logger = logging.getLogger(__name__)


@dataclass
class LeiaTrainingArguments(TrainingArguments):
    model_name_or_path: str | None = field(default=None)
    use_flash_attention_2: bool = field(default=False)

    wikipedia_dataset_dir: str | None = field(default=None)
    text_dataset_path: str | None = field(default=None)
    text_dataset_name: str | None = field(default=None)
    text_dataset_sampling_prob: float = field(default=0.0)

    trans_insertion_strategy: str = field(default="none")
    trans_insertion_prob: float = field(default=1.0)
    trans_insertion_prob_decay: bool = field(default=False)
    trans_token_loss_weight: float = field(default=1.0)

    max_length: int = field(default=1024)
    min_lr_ratio: float = field(default=0.0)

    eval_tasks: str | None = field(default=None)
    max_eval_samples_for_tasks: int | None = field(default=None)
    num_fewshot_samples_for_tasks: str | None = field(default=None)
    use_dynamic_generation_length: bool = field(default=True)
    eval_at_first_step: bool = field(default=False)


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


def main():
    parser = HfArgumentParser(LeiaTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    if args.local_rank == 0:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    datasets.utils.logging.set_verbosity(logging.ERROR)
    transformers.utils.logging.set_verbosity(logging.WARNING)

    logger.info(f"Arguments: {args}")

    set_seed(args.seed)

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    def custom_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        trans_token_mask: torch.Tensor | None = None,
        # position_ids: torch.Tensor | None = None,
        # past_key_values: list[torch.Tensor] | None = None,
        # inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs
        # use_cache: bool | None = None,
        # output_attentions: bool | None = None,
        # output_hidden_states: bool | None = None,
        # return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            **kwargs
            # return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            if args.trans_token_loss_weight != 1.0 and trans_token_mask is not None:
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits, shift_labels)
                loss_weights = torch.ones_like(loss)
                loss_weights += trans_token_mask.type_as(loss).view(-1) * (args.trans_token_loss_weight - 1.0)
                loss_weights *= (shift_labels != -100).float()
                loss *= loss_weights
                loss = loss.mean()

            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = custom_forward

    if args.local_rank == 0:
        logger.info(f"Model: {model}")

    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["<trans>", "</trans>"]})
    assert num_new_tokens == 2
    model.resize_token_embeddings(len(tokenizer))

    embeddings_data = model.get_input_embeddings().weight.data
    embeddings_avg = embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
    embeddings_data[-num_new_tokens:] = embeddings_avg

    logger.info("Loading datasets...")
    wikipedia_datasets = []
    for wikipedia_dataset_dir in args.wikipedia_dataset_dir.split(","):
        wikipedia_datasets.append(load_from_disk(wikipedia_dataset_dir))
    wikipedia_dataset = concatenate_datasets(wikipedia_datasets)
    wikipedia_dataset_size = len(wikipedia_dataset)
    wikipedia_dataset = wikipedia_dataset.to_iterable_dataset(num_shards=1024)

    if args.text_dataset_path:
        text_dataset = load_dataset(args.text_dataset_path, name=args.text_dataset_name, split="train", streaming=True)

        text_dataset = text_dataset.map(
            lambda texts: {"input_ids": tokenizer(texts)["input_ids"]},
            input_columns=["text"],
            remove_columns=list(next(iter(text_dataset)).keys()),
            batched=True,
            features=datasets.Features({"input_ids": datasets.Sequence(datasets.Value(dtype="int16"))}),
        )

        train_dataset = interleave_datasets(
            [wikipedia_dataset, text_dataset],
            probabilities=[
                1.0 - args.text_dataset_sampling_prob,
                args.text_dataset_sampling_prob,
            ],
            stopping_strategy="first_exhausted",
            seed=args.seed,
        )
        # We assume that the text dataset is much larger than the Wikipedia dataset
        train_dataset_size = int(wikipedia_dataset_size / (1.0 - args.text_dataset_sampling_prob))

    else:
        train_dataset = wikipedia_dataset
        train_dataset_size = wikipedia_dataset_size

    max_num_examples = (
        args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
    )
    train_cl_dataset = LeiaConstantLengthDataset(
        train_dataset,
        dataset_size=train_dataset_size,
        max_length=args.max_length,
        max_num_examples=max_num_examples,
        trans_start_token_id=tokenizer.vocab["<trans>"],
        trans_end_token_id=tokenizer.vocab["</trans>"],
        trans_insertion_prob=args.trans_insertion_prob,
        trans_insertion_prob_decay=args.trans_insertion_prob_decay,
        trans_insertion_strategy=args.trans_insertion_strategy,
        shuffle=True,
        seed=args.seed,
    )

    data_collator = LeiaDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    eval_tasks = []
    if args.eval_tasks is not None:
        eval_tasks = args.eval_tasks.split(",")

    num_fewshot_samples_for_tasks = None
    if args.num_fewshot_samples_for_tasks is not None:
        num_fewshot_samples_for_tasks = [int(n) for n in args.num_fewshot_samples_for_tasks.split(",")]

    trainer = LeiaTrainer(
        model=model,
        args=args,
        train_dataset=train_cl_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_tasks=eval_tasks,
        num_fewshot_samples_for_tasks=num_fewshot_samples_for_tasks,
        eval_task_kwargs={
            "max_length": args.max_length,
            "max_samples": args.max_eval_samples_for_tasks,
        },
        eval_generation_task_kwargs={
            "use_dynamic_generation_length": args.use_dynamic_generation_length,
        },
    )
    if args.eval_at_first_step:
        trainer.add_callback(EvaluateFirstStepCallback())

    if args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
