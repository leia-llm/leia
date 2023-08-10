import argparse
import logging
import os

import torch
from tqdm import tqdm

from leia.utils import load_tsv_mapping


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)
    reverse_wikidata_id_mapping = {v: k for k, v in wikidata_id_mapping.items()}
    entity_vocab = load_tsv_mapping(args.entity_vocab_file, int)
    reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}

    embedding_files = args.embedding_files.split(",")
    prev_token_embeddings = {}
    last_token_embeddings = {}
    for embedding_file in tqdm(embedding_files, desc="Loading embeddings"):
        embeddings_dict = torch.load(embedding_file, map_location="cpu")
        prev_token_embeddings[embedding_file] = embeddings_dict["prev_token_embeddings"]
        prev_token_embeddings[embedding_file] /= torch.norm(
            prev_token_embeddings[embedding_file], dim=1, keepdim=True
        ).clamp(min=0.0001)
        last_token_embeddings[embedding_file] = embeddings_dict["last_token_embeddings"]
        last_token_embeddings[embedding_file] /= torch.norm(
            last_token_embeddings[embedding_file], dim=1, keepdim=True
        ).clamp(min=0.0001)
        del embeddings_dict

    while True:
        try:
            query_entity_title = input("Entity title: ")
        except EOFError:
            continue
        if query_entity_title not in wikidata_id_mapping:
            print(f"Entity {query_entity_title} is not found in the mapping")
            continue
        query_wikidata_id = wikidata_id_mapping[query_entity_title]
        if query_wikidata_id not in entity_vocab:
            print(f"Entity {query_entity_title} ({query_wikidata_id}) is not found in the vocabulary")
            continue
        query_entity_index = entity_vocab[query_wikidata_id]

        for embedding_file in embedding_files:
            for label, embeddings in (
                ("prev token", prev_token_embeddings[embedding_file]),
                ("last token", last_token_embeddings[embedding_file]),
            ):
                print(f"Nearest entities in {label} embeddings ({os.path.basename(embedding_file)}):")
                scores = torch.matmul(embeddings, embeddings[query_entity_index])
                topk_scores, topk_indices = scores.topk(args.k + 1, largest=True, sorted=True)
                for n, (score, nearest_entity_index) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist())):
                    if nearest_entity_index == query_entity_index:
                        continue
                    nearest_entity_title = reverse_wikidata_id_mapping[reverse_entity_vocab[nearest_entity_index]]
                    print(f"{n}. {nearest_entity_title} (score: {score:.3f})", end="  ")
                print("")
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikidata_id_file", type=str, required=True)
    parser.add_argument("--entity_vocab_file", type=str, required=True)
    parser.add_argument("--embedding_files", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    main(args)
