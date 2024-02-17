import argparse
import bz2
import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


def _extract_wikidata_sitelinks(dump_file: str, languages: list[str]) -> dict[str, list[tuple[str, str]]]:
    language_set = frozenset(languages)
    sitelinks: dict[str, list[tuple[str, str]]] = defaultdict(list)

    with bz2.open(dump_file, "rt") as input_file:
        for n, line in enumerate(input_file):
            if n % 100000 == 0 and n != 0:
                logger.info(f"Processed {n} lines")

            line = line.rstrip()
            if line in ("[", "]"):
                continue

            if line[-1] == ",":
                line = line[:-1]
            item = json.loads(line)
            if item["type"] != "item":
                continue

            wikidata_id = item["id"]
            for link_item in item["sitelinks"].values():
                site = link_item["site"]
                if not site.endswith("wiki"):
                    continue
                language = site[:-4]
                if language_set and language not in language_set:
                    continue

                sitelinks[language].append((link_item["title"], wikidata_id))

    return sitelinks


def main(args: argparse.Namespace) -> None:
    languages = args.languages.split(",")
    for language, sitelinks in _extract_wikidata_sitelinks(args.dump_file, languages=languages).items():
        with open(os.path.join(args.output_dir, f"{language}-wikidata-ids.tsv"), "w") as output_file:
            for title, wikidata_id in sitelinks:
                output_file.write(f"{title}\t{wikidata_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
