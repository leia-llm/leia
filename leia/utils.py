import math
from functools import partial


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def normalize_wikipedia_title(title: str) -> str:
    return (title[0].upper() + title[1:]).replace("_", " ")


def load_tsv_mapping(mapping_file: str, value_type: type = str) -> dict:
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            key, value = line.rstrip().split("\t")[:2]
            if not isinstance(value, value_type):
                value = value_type(value)
            mapping[key] = value

    return mapping


def get_cosine_schedule_with_warmup_and_min_lr_ratio(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float, last_epoch: int = -1
):
    def lr_lambda_func(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    func = partial(
        lr_lambda_func,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, func, last_epoch)
