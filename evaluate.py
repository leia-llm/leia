import argparse
import json
import logging
import os
from pprint import pprint

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from leia.tasks import get_task

logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=args.use_flash_attention_2,
    )
    max_length = getattr(model.config, "max_position_embeddings", None) if args.max_length is None else args.max_length
    if max_length is None:
        max_length = 2048

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "qwen" in args.model_name_or_path.lower():
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"

    tasks = args.task.split(",")
    num_fewshot_samples = [int(x) for x in args.num_fewshot_samples.split(",")]
    assert len(tasks) == len(
        num_fewshot_samples
    ), "The length of tasks and the length of num_fewshot_samples must be the same."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    for task_name, num_samples in zip(tasks, num_fewshot_samples):
        task_cls = get_task(task_name)
        task = task_cls(
            model=model,
            accelerator=accelerator,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=max_length,
            num_fewshot_samples=num_samples,
            max_samples=args.max_samples,
        )
        result = task.run()
        if accelerator.is_main_process:
            print(task_name, result.metrics)

            if args.output_dir is not None:
                with open(os.path.join(args.output_dir, f"{task_name}_metrics.json"), "w") as f:
                    json.dump(result.metrics, f, indent=2)
                with open(os.path.join(args.output_dir, f"{task_name}_predictions.jsonl"), "w") as f:
                    for example, prediction in zip(result.examples, result.predictions):
                        f.write(f'{json.dumps({"example": example, "prediction": prediction}, ensure_ascii=False)}\n')

            all_metrics[task_name] = result.metrics

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        pprint(all_metrics)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_fewshot_samples", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(args)
