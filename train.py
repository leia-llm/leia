import logging
from dataclasses import dataclass, field

import datasets
import transformers
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from leia.data import LeiaConstantLengthDataset, LeiaDataCollator

logger = logging.getLogger(__name__)


@dataclass
class LeiaTrainingArguments(TrainingArguments):
    model_name_or_path: str | None = field(default=None)
    use_flash_attention_2: bool = field(default=True)

    wikipedia_dataset_dir: str | None = field(default=None)

    entity_name_insertion_strategy: str = field(default="right")
    entity_name_insertion_prob: float = field(default=0.5)
    disable_entity_name_token_loss: bool = field(default=False)
    no_separator_tokens: bool = field(default=False)

    max_length: int = field(default=2048)


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

    if args.local_rank == 0:
        logger.info(f"Model: {model}")

    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["<translate>", "</translate>"]})
    assert num_new_tokens == 2
    model.resize_token_embeddings(len(tokenizer))

    embeddings_data = model.get_input_embeddings().weight.data
    embeddings_avg = embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
    embeddings_data[-num_new_tokens:] = embeddings_avg

    logger.info("Loading datasets...")
    wikipedia_dataset = load_from_disk(args.wikipedia_dataset_dir)

    max_num_examples = (
        args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
    )
    train_cl_dataset = LeiaConstantLengthDataset(
        wikipedia_dataset,
        dataset_size=len(wikipedia_dataset),
        max_length=args.max_length,
        max_num_examples=max_num_examples,
        entity_name_start_token_id=tokenizer.vocab["<translate>"],
        entity_name_end_token_id=tokenizer.vocab["</translate>"],
        entity_name_insertion_prob=args.entity_name_insertion_prob,
        entity_name_insertion_strategy=args.entity_name_insertion_strategy,
        no_separator_tokens=args.no_separator_tokens,
        shuffle=True,
        seed=args.seed,
    )

    data_collator = LeiaDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        disable_entity_name_token_loss=args.disable_entity_name_token_loss,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_cl_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
