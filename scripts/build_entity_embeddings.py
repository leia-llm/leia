import argparse
import logging
import os
import time

import datasets
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from tqdm import tqdm

from leia.data import LeiaDataCollator
from leia.utils import load_tsv_mapping


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, use_flash_attention_2=args.use_flash_attention_2, torch_dtype=torch.float16
    ).eval()
    hidden_size = model.config.hidden_size

    if args.layers is not None:
        layers = [int(layer) for layer in args.layers.split(",")]
    else:
        layers = list(range(int(model.config.num_hidden_layers / 2), model.config.num_hidden_layers))

    accelerator.print(f"target layers: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "llama" in args.model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id

    entity_vocab_file = os.path.join(args.wikipedia_dataset_dir, "entity_vocab.tsv")
    entity_vocab = load_tsv_mapping(entity_vocab_file, int)
    entity_padding_index = entity_vocab["<pad>"]
    accelerator.print(f"entity vocab size: {len(entity_vocab)}")

    dataset = datasets.load_from_disk(args.wikipedia_dataset_dir)
    accelerator.print(f"dataset size: {len(dataset)}")

    if args.start_index is not None or args.end_index is not None:
        start_index = args.start_index
        if start_index is None:
            start_index = 0
        end_index = args.end_index
        if end_index is None:
            end_index = len(dataset)
        accelerator.print(f"start index: {args.start_index}")
        accelerator.print(f"end index: {end_index}")
        dataset = dataset.select(range(args.start_index, end_index))

    accelerator.print(f"target dataset size: {len(dataset)}")

    dataset = dataset.with_format("torch")

    sampler = LengthGroupedSampler(args.batch_size, dataset=dataset)
    collator = LeiaDataCollator(tokenizer=tokenizer, entity_vocab=entity_vocab, padding="longest")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_dataloader_workers,
        drop_last=True,
        pin_memory=True,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    if accelerator.is_main_process:
        all_prev_token_embeddings = torch.zeros(len(layers), len(entity_vocab), hidden_size, dtype=torch.float32)
        all_last_token_embeddings = torch.zeros(len(layers), len(entity_vocab), hidden_size, dtype=torch.float32)
        entity_counter = torch.zeros(len(entity_vocab), dtype=torch.long)

        entity_counter_file = os.path.join(args.output_dir, "entity_counter.pt")
        if os.path.exists(entity_counter_file):
            accelerator.print(f"Loading existing embeddings from {args.output_dir}...")
            entity_counter = torch.load(entity_counter_file)
            for n, layer_index in enumerate(layers):
                embedding_file = os.path.join(args.output_dir, f"layer_{layer_index}.pt")
                accelerator.print(f"Loading {embedding_file}...")
                assert os.path.exists(embedding_file), f"{embedding_file} does not exist."

                embedding_dict = torch.load(embedding_file)
                all_prev_token_embeddings[n] = embedding_dict["prev_token_embeddings"]
                all_last_token_embeddings[n] = embedding_dict["last_token_embeddings"]

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.inference_mode():
            hidden_states = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True
            ).hidden_states
            # shape: [batch, layer, sequence, hidden]
            hidden_states = torch.stack([hidden_states[n] for n in layers], dim=1)

        # shape: [batch, layer, entity, hidden]
        prev_token_embeddings = torch.gather(
            hidden_states,
            2,
            batch["entity_prev_token_positions"][:, None, :, None].expand(-1, len(layers), -1, hidden_size),
        )
        last_token_embeddings = torch.gather(
            hidden_states,
            2,
            batch["entity_last_token_positions"][:, None, :, None].expand(-1, len(layers), -1, hidden_size),
        )
        prev_token_embeddings, last_token_embeddings = accelerator.pad_across_processes(
            [prev_token_embeddings, last_token_embeddings], dim=2, pad_index=0
        )
        prev_token_embeddings, last_token_embeddings = accelerator.gather(
            [prev_token_embeddings, last_token_embeddings]
        )

        # shape: [batch, entity]
        entity_ids = accelerator.pad_across_processes(batch["entity_ids"], dim=1, pad_index=entity_padding_index)
        entity_ids = accelerator.gather(entity_ids)

        if accelerator.is_main_process:
            # each instance is processed separately because the same entity can appear multiple times in entity_ids
            for index in range(args.batch_size):
                target_entity_ids = entity_ids[index]
                prev_token_embeddings[index] *= (target_entity_ids != entity_padding_index).float()[None, :, None]
                last_token_embeddings[index] *= (target_entity_ids != entity_padding_index).float()[None, :, None]

                target_entity_ids = target_entity_ids.to("cpu")
                all_prev_token_embeddings[:, target_entity_ids] += prev_token_embeddings[index].to("cpu")
                all_last_token_embeddings[:, target_entity_ids] += last_token_embeddings[index].to("cpu")
                entity_counter[target_entity_ids] += 1

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        for n, layer_index in enumerate(layers):
            output_dict = {
                "prev_token_embeddings": all_prev_token_embeddings[n].clone(),
                "last_token_embeddings": all_last_token_embeddings[n].clone(),
            }
            output_file = os.path.join(args.output_dir, f"layer_{layer_index}.pt")
            torch.save(output_dict, output_file)

        entity_counter_file = os.path.join(args.output_dir, "entity_counter.pt")
        torch.save(entity_counter, entity_counter_file)
        print(f"Saved embeddings to {args.output_dir}")
    else:
        time.sleep(600)  # saving embeddings can take longer than the sync timeout
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_dataset_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--num_dataloader_workers", type=int, default=4)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()

    main(args)
