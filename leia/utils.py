from typing import Any

import torch


def normalize_wikipedia_title(title: str) -> str:
    return (title[0].upper() + title[1:]).replace("_", " ")


def load_tsv_mapping(mapping_file: str, value_type: type = str) -> dict[str, Any]:
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            key, value = line.rstrip().split("\t")[:2]
            if not isinstance(value, value_type):
                value = value_type(value)
            mapping[key] = value

    return mapping


def find_all_linear_names(model: torch.nn.Module) -> list[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "decoder" in lora_module_names:
        lora_module_names.remove("decoder")

    return list(lora_module_names)
