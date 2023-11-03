import logging
import multiprocessing
import os
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import interleave_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from leia.data import LeiaConstantLengthDataset, LeiaDataCollator
from leia.model import LeiaLlamaConfig, LeiaLlamaForCausalLM
from leia.trainer import LeiaTrainer
from leia.utils import find_all_linear_names, load_tsv_mapping

logger = logging.getLogger(__name__)


@dataclass
class LeiaTrainingArguments(TrainingArguments):
    model_name_or_path: str | None = field(default=None)
    entity_embedding_dir: str | None = field(default=None)
    wikipedia_dataset_dir: str | None = field(default=None)
    text_dataset_path: str | None = field(default=None)
    text_dataset_name: str | None = field(default=None)
    text_dataset_sampling_prob: float | None = field(default=None)
    max_eval_samples_for_perplexity: int | None = field(default=None)
    max_eval_samples_for_tasks: int | None = field(default=None)
    eval_tasks: str | None = field(default=None)
    max_length: int = field(default=1024)
    max_entity_length: int = field(default=32)
    entity_vocab_size: int | None = field(default=None)
    do_language_modeling: bool = field(default=True)
    similarity_function: str = field(default="cosine")
    temperature: float = field(default=0.1)
    layer_index: int = field(default=31)
    use_entity_decoder_activation: bool = field(default=False)
    num_preprocess_workers: int = field(default=multiprocessing.cpu_count())
    use_flash_attention_2: bool = field(default=False)
    use_entity_prev_token_prediction: bool = field(default=True)
    use_entity_last_token_prediction: bool = field(default=True)
    use_dynamic_generation_length: bool = field(default=True)

    use_lora: bool = field(default=False)
    # full_finetune: bool = field(default=False)
    # bits: int = field(default=8)
    lora_r: int = field(default=64)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.0)
    lora_train_embedding: bool = field(default=False)


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

    logger.info("Loading datasets...")
    wikipedia_dataset = load_from_disk(args.wikipedia_dataset_dir)
    estimated_dataset_size = len(wikipedia_dataset)
    wikipedia_dataset = wikipedia_dataset.to_iterable_dataset(num_shards=1024)

    train_source_dataset = wikipedia_dataset
    entity_vocab_file = os.path.join(args.wikipedia_dataset_dir, "entity_vocab.tsv")
    entity_vocab = load_tsv_mapping(entity_vocab_file, int)
    if args.entity_vocab_size is not None:
        entity_vocab = {k: v for k, v in entity_vocab.items() if v < args.entity_vocab_size}

    eval_source_dataset = None
    if args.text_dataset_path is not None:
        text_dataset = load_dataset(args.text_dataset_path, name=args.text_dataset_name, split="train", streaming=True)

        text_dataset = text_dataset.map(
            lambda texts: {"input_ids": tokenizer(texts)["input_ids"]},
            input_columns=["text"],
            remove_columns=list(next(iter(text_dataset)).keys()),
            batched=True,
            features=datasets.Features({"input_ids": datasets.Sequence(datasets.Value(dtype="int16"))}),
        )
        if args.max_eval_samples_for_perplexity is not None:
            dataset_dict = text_dataset.train_test_split(
                test_size=args.max_eval_samples_for_perplexity, shuffle=True, seed=args.seed
            )
            eval_source_dataset = dataset_dict["test"]
            text_dataset = dataset_dict["train"]

        if args.text_dataset_sampling_prob is not None and args.text_dataset_sampling_prob > 0.0:
            train_source_dataset = interleave_datasets(
                [train_source_dataset, text_dataset],
                probabilities=[
                    1.0 - args.text_dataset_sampling_prob,
                    args.text_dataset_sampling_prob,
                ],
                stopping_strategy="first_exhausted",
                seed=args.seed,
            )
            # We assume that the text dataset is much larger than the Wikipedia dataset
            estimated_dataset_size = int(estimated_dataset_size / (1.0 - args.text_dataset_sampling_prob))

    train_dataset = LeiaConstantLengthDataset(
        train_source_dataset,
        max_length=args.max_length,
        max_entity_length=args.max_entity_length,
        entity_vocab_size=args.entity_vocab_size,
        infinite=True,
        shuffle=True,
        seed=args.seed,
        dataset_size=estimated_dataset_size,
    )
    if eval_source_dataset is None:
        eval_dataset = None
    else:
        eval_dataset = LeiaConstantLengthDataset(
            eval_source_dataset,
            max_length=args.max_length,
            max_entity_length=args.max_entity_length,
            entity_vocab_size=args.entity_vocab_size,
            infinite=False,
            shuffle=False,
        )

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "llama" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = LeiaLlamaConfig.from_pretrained(
        args.model_name_or_path,
        entity_vocab_size=args.entity_vocab_size,
        similarity_function=args.similarity_function,
        temperature=args.temperature,
        layer_index=args.layer_index,
        use_entity_decoder_activation=args.use_entity_decoder_activation,
        use_entity_prev_token_prediction=args.use_entity_prev_token_prediction,
        use_entity_last_token_prediction=args.use_entity_last_token_prediction,
        use_cache=not args.gradient_checkpointing,
    )

    model = LeiaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    if args.use_lora:
        modules_to_save = []
        if args.lora_train_embedding:
            modules_to_save = ["lm_head", "embed_tokens"]

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_all_linear_names(model),
            modules_to_save=modules_to_save,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    logger.info("Initialize entity embeddings...")
    entity_embedding_file = os.path.join(args.entity_embedding_dir, f"layer_{args.layer_index}.pt")
    entity_embedding = torch.load(entity_embedding_file)
    prev_token_embeddings = entity_embedding["prev_token_embeddings"][: args.entity_vocab_size]
    last_token_embeddings = entity_embedding["last_token_embeddings"][: args.entity_vocab_size]
    if args.similarity_function == "cosine":
        prev_token_embeddings = torch.nn.functional.normalize(prev_token_embeddings, dim=-1)
        last_token_embeddings = torch.nn.functional.normalize(last_token_embeddings, dim=-1)
    elif args.similarity_function == "dot":
        entity_counter_file = os.path.join(args.entity_embedding_dir, "entity_counter.pt")
        entity_counter = torch.load(entity_counter_file)
        entity_counter = entity_counter[: args.entity_vocab_size]
        prev_token_embeddings = prev_token_embeddings / entity_counter.float().clamp(1e-12).unsqueeze(-1)
        last_token_embeddings = last_token_embeddings / entity_counter.float().clamp(1e-12).unsqueeze(-1)
        del entity_counter

    model.prev_token_head.decoder.decoder.weight.data.copy_(prev_token_embeddings)
    model.last_token_head.decoder.decoder.weight.data.copy_(last_token_embeddings)
    model.prev_token_head.decoder.decoder.weight.requires_grad = False
    model.last_token_head.decoder.decoder.weight.requires_grad = False
    model.prev_token_head.dense.weight.data.copy_(torch.eye(config.hidden_size))
    model.last_token_head.dense.weight.data.copy_(torch.eye(config.hidden_size))

    if args.local_rank == 0:
        logger.info(f"Model: {model}")

    del entity_embedding, prev_token_embeddings, last_token_embeddings

    data_collator = LeiaDataCollator(
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        max_length=args.max_length,
        max_entity_length=args.max_entity_length,
        do_language_modeling=args.do_language_modeling,
    )

    eval_tasks = []
    if args.eval_tasks is not None:
        eval_tasks = args.eval_tasks.split(",")

    trainer = LeiaTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_tasks=eval_tasks,
        eval_task_kwargs={
            "max_length": args.max_length,
            "max_samples": args.max_eval_samples_for_tasks,
        },
        eval_generation_task_kwargs={
            "use_dynamic_generation_length": args.use_dynamic_generation_length,
        }
        # callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )

    checkpoint = None
    if args.resume_from_checkpoint is not None:
        # checkpoint = args.resume_from_checkpoint
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.train()

        trainer.save_state()
        trainer.save_model()
        # if model_args.use_peft:
        #     model.save_pretrained(training_args.output_dir)
        # else:
        #     trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    main()
