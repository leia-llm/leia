import argparse
import math

import datasets
from tqdm import tqdm

from leia.utils import load_tsv_mapping

entity_text_mapping = None


def main(args: argparse.Namespace) -> None:
    wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)
    entity_text_mapping = {v: k for k, v in wikidata_id_mapping.items()}
    del wikidata_id_mapping

    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    total_num_anchors = 0
    valid_num_anchors = 0
    with tqdm(total=math.ceil(len(dataset) / 1000), leave=False) as pbar:
        for examples in dataset.select_columns(["anchors"]).iter(batch_size=1000):
            for anchors in examples["anchors"]:
                for anchor in anchors:
                    total_num_anchors += 1
                    wikidata_id = anchor["wikidata_id"]
                    if wikidata_id is not None and wikidata_id in entity_text_mapping:
                        valid_num_anchors += 1
            pbar.update()

    print("#anchors:", total_num_anchors)
    print("#valid anchors:", valid_num_anchors)
    print("ratio:", valid_num_anchors / total_num_anchors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_dir", type=str, required=True)
    parser.add_argument("--wikidata_id_file", type=str, required=True)
    args = parser.parse_args()

    main(args)
