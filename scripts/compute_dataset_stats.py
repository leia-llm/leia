import argparse
import math

import datasets
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    dataset = datasets.load_from_disk(args.wikipedia_dataset_dir)
    token_count = 0
    entity_count = 0
    with tqdm(total=math.ceil(len(dataset) / 1000), leave=False) as pbar:
        for examples in dataset.select_columns(["input_ids", "entity_start_positions"]).iter(batch_size=1000):
            token_count += sum(len(input_ids) for input_ids in examples["input_ids"])
            entity_count += sum(len(entity_ids) for entity_ids in examples["entity_start_positions"])
            pbar.update()

    print("#articles:", len(dataset))
    print("#tokens:", token_count)
    print("#entities:", entity_count)
    print("#tokens/article:", token_count / len(dataset))
    print("#entities/article:", entity_count / len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_dataset_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
