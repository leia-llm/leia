import logging
import os
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from leia.data import LeiaConstantLengthDataset, LeiaDataCollator
from leia.model import LeiaLlamaConfig, LeiaLlamaForCausalLM
from leia.trainer import LeiaTrainer
from leia.utils import load_tsv_mapping

logger = logging.getLogger(__name__)


@dataclass
class LeiaTrainingArguments(TrainingArguments):
    model_name_or_path: str | None = field(default=None)
    use_flash_attention_2: bool = field(default=False)

    entity_embedding_dir: str | None = field(default=None)
    wikipedia_dataset_dir: str | None = field(default=None)
    text_dataset_path: str | None = field(default=None)
    text_dataset_name: str | None = field(default=None)
    text_dataset_sampling_prob: float = field(default=0.0)

    max_length: int = field(default=1024)
    max_entity_length: int = field(default=32)
    entity_vocab_size: int | None = field(default=None)

    layer_index: int = field(default=31)
    similarity_function: str = field(default="cosine")
    temperature: float = field(default=0.1)
    use_entity_prev_token_prediction: bool = field(default=True)
    use_entity_last_token_prediction: bool = field(default=True)
    use_entity_decoder_activation: bool = field(default=False)

    eval_tasks: str | None = field(default=None)
    max_eval_samples_for_tasks: int | None = field(default=None)
    num_fewshot_samples_for_tasks: int = field(default=0)
    use_dynamic_generation_length: bool = field(default=True)

    train_entity_dense_only: bool = field(default=False)
    num_train_wikipedia_samples: int | None = field(default=None)
    num_eval_wikipedia_samples: int | None = field(default=None)
    skip_wikipedia_samples: int | None = field(default=None)
    load_entity_dense_weights: bool = field(default=False)


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
    train_dataset = load_from_disk(args.wikipedia_dataset_dir)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    eval_dataset = None
    if args.num_eval_wikipedia_samples is not None:
        eval_dataset = train_dataset.select(range(args.num_eval_wikipedia_samples))
        train_dataset = train_dataset.select(range(args.num_eval_wikipedia_samples, len(train_dataset)))

    if args.skip_wikipedia_samples is not None:
        train_dataset = train_dataset.select(range(args.skip_wikipedia_samples, len(train_dataset)))
    if args.num_train_wikipedia_samples is not None:
        train_dataset = train_dataset.select(range(args.num_train_wikipedia_samples))

    train_dataset_size = len(train_dataset)
    train_dataset = train_dataset.to_iterable_dataset(num_shards=1024)

    entity_vocab_file = os.path.join(args.wikipedia_dataset_dir, "entity_vocab.tsv")
    entity_vocab = load_tsv_mapping(entity_vocab_file, int)
    if args.entity_vocab_size is not None:
        entity_vocab = {k: v for k, v in entity_vocab.items() if v < args.entity_vocab_size}

    if args.text_dataset_path:
        assert (
            not args.train_entity_dense_only
        ), "text_dataset_path should not be set if train_entity_dense_only is enabled"

        text_dataset = load_dataset(args.text_dataset_path, name=args.text_dataset_name, split="train", streaming=True)

        text_dataset = text_dataset.map(
            lambda texts: {"input_ids": tokenizer(texts)["input_ids"]},
            input_columns=["text"],
            remove_columns=list(next(iter(text_dataset)).keys()),
            batched=True,
            features=datasets.Features({"input_ids": datasets.Sequence(datasets.Value(dtype="int16"))}),
        )

        if args.text_dataset_sampling_prob > 0.0:
            train_dataset = interleave_datasets(
                [train_dataset, text_dataset],
                probabilities=[
                    1.0 - args.text_dataset_sampling_prob,
                    args.text_dataset_sampling_prob,
                ],
                stopping_strategy="first_exhausted",
                seed=args.seed,
            )
            # We assume that the text dataset is much larger than the Wikipedia dataset
            train_dataset_size = int(train_dataset_size / (1.0 - args.text_dataset_sampling_prob))

    train_cl_dataset = LeiaConstantLengthDataset(
        train_dataset,
        max_length=args.max_length,
        max_entity_length=args.max_entity_length,
        entity_vocab_size=args.entity_vocab_size,
        infinite=True,
        shuffle=True,
        seed=args.seed,
        dataset_size=train_dataset_size,
    )
    eval_cl_dataset = None
    if eval_dataset is not None:
        eval_cl_dataset = LeiaConstantLengthDataset(
            eval_dataset,
            max_length=args.max_length,
            max_entity_length=args.max_entity_length,
            entity_vocab_size=args.entity_vocab_size,
            infinite=False,
            shuffle=False,
        )

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
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

    logger.info("Initializing entity embeddings...")
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
    if args.load_entity_dense_weights:
        entity_dense_weights_file = os.path.join(args.output_dir, "entity_dense_weights.pt")
        if not os.path.exists(entity_dense_weights_file):
            raise RuntimeError("entity_dense_weights.pt does not exist")
        entity_dense_weights = torch.load(entity_dense_weights_file, map_location="cpu")
        model.prev_token_head.dense.weight.data.copy_(entity_dense_weights["prev_token_head.dense.weight"])
        model.last_token_head.dense.weight.data.copy_(entity_dense_weights["last_token_head.dense.weight"])
    else:
        model.prev_token_head.dense.weight.data.copy_(torch.eye(config.hidden_size))
        model.last_token_head.dense.weight.data.copy_(torch.eye(config.hidden_size))

    if args.train_entity_dense_only:
        for param in model.parameters():
            param.requires_grad = False
        model.prev_token_head.dense.weight.requires_grad = True
        model.last_token_head.dense.weight.requires_grad = True
    else:
        model.prev_token_head.decoder.decoder.weight.requires_grad = False
        model.last_token_head.decoder.decoder.weight.requires_grad = False

    if args.local_rank == 0:
        logger.info(f"Model: {model}")

    del entity_embedding, prev_token_embeddings, last_token_embeddings

    data_collator = LeiaDataCollator(
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        max_length=args.max_length,
        max_entity_length=args.max_entity_length,
        do_language_modeling=True,
    )

    eval_tasks = []
    if args.eval_tasks is not None:
        eval_tasks = args.eval_tasks.split(",")

    trainer = LeiaTrainer(
        model=model,
        args=args,
        train_dataset=train_cl_dataset,
        eval_dataset=eval_cl_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_tasks=eval_tasks,
        eval_task_kwargs={
            "max_length": args.max_length,
            "max_samples": args.max_eval_samples_for_tasks,
            "num_fewshot_samples": args.num_fewshot_samples_for_tasks,
        },
        eval_generation_task_kwargs={
            "use_dynamic_generation_length": args.use_dynamic_generation_length,
        },
    )

    if args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    if args.train_entity_dense_only:
        state_dict = trainer.accelerator.get_state_dict(trainer.model_wrapped)
        if args.process_index == 0:
            torch.save(
                {
                    key: value
                    for key, value in state_dict.items()
                    if key in ("prev_token_head.dense.weight", "last_token_head.dense.weight")
                },
                os.path.join(args.output_dir, "entity_dense_weights.pt"),
            )
    else:
        trainer.save_state()
        trainer.save_model()


if __name__ == "__main__":
    main()
