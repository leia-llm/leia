import argparse
import html
import glob
import logging
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

import bs4
import datasets

from leia.utils import load_tsv_mapping, normalize_wikipedia_title

logger = logging.getLogger(__name__)


def _generate_articles(wikiextractor_output_dir: str):
    for article_file in glob.glob("**/wiki_*", root_dir=wikiextractor_output_dir, recursive=True):
        with open(os.path.join(wikiextractor_output_dir, article_file), "r") as f:
            for line in f:
                if line.startswith("<doc id="):
                    metadata = ET.fromstring("{}{}".format(line, "</doc>")).attrib
                    title = metadata["title"]
                    url = metadata["url"]
                    language = url.split(".")[0].split("//")[1]
                    html = ""
                    line_num = 0

                elif line.startswith("</doc>"):
                    if not re.match(r"^\s*$", html):
                        doc = {"title": title, "language": language, "html": html}
                        yield doc

                else:
                    # first two lines are an article title and an empty line
                    # https://github.com/attardi/wikiextractor/blob/8f1b434a80608e1e313d38d263ed7c79c9ee75a9/wikiextractor/extract.py#L991
                    if line_num > 1:
                        html += line
                    line_num += 1


def _parse_article(
    examples: dict[str, list[str]],
    redirects: dict[str, str],
    wikidata_id_mapping: dict[str, str],
) -> dict[str, list]:
    titles = examples["title"]
    html_texts = examples["html"]
    languages = examples["language"]
    texts = []
    anchors_list: list[list[dict]] = []
    for title, html_text, language in zip(titles, html_texts, languages):
        try:
            soup = bs4.BeautifulSoup(html_text, "html.parser")
        except Exception as e:
            logger.warning(f"BeautifulSoup error on parsing {title}: {e}")
            texts.append("")
            anchors_list.append([])
            continue

        text = ""
        anchors: list[dict] = []
        for span in soup:
            if isinstance(span, bs4.element.Tag):
                span_text = span.get_text()
                href = span.get("href", None)
                if href:
                    href = urllib.parse.unquote(href)
                    href = html.unescape(href)
                    if "#" not in href:
                        href_title = _get_title_from_href(href, language)
                        if href_title:
                            href_title = redirects.get(href_title, href_title)
                            wikidata_id = wikidata_id_mapping.get(href_title, None)

                            anchors.append(
                                {
                                    "start": len(text),
                                    "end": len(text) + len(span_text),
                                    "title": href_title,
                                    "wikidata_id": wikidata_id,
                                }
                            )
                text += span_text
            else:
                text += str(span)

        texts.append(text)
        anchors_list.append(anchors)

    return {"text": texts, "anchors": anchors_list}


def _get_title_from_href(href: str, language: str) -> str | None:
    if href.startswith(f"https://{language}.wikipedia.org/wiki/"):
        href = href[len(f"https://{language}.wikipedia.org/wiki/") :]
    elif href.startswith("http://") or href.startswith("https://"):
        return None
    elif href.startswith("/wiki/"):
        href = href[len("/wiki/") :]
    elif href.startswith(f"w:{language}:"):
        href = href[len(f"w:{language}:") :]
    elif href.startswith("w:") and ":" not in href[len("w:") :]:
        href = href[len("w:") :]
    elif href.startswith(f":{language}:"):
        href = href[len(f":{language}:") :]

    if not href:
        return None

    href = normalize_wikipedia_title(href)
    return href


def main(args: argparse.Namespace) -> None:
    redirects = load_tsv_mapping(args.redirect_file)
    wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)

    dataset = datasets.IterableDataset.from_generator(
        _generate_articles, gen_kwargs={"wikiextractor_output_dir": args.wikiextractor_output_dir}
    )
    dataset = dataset.map(
        _parse_article,
        fn_kwargs={"redirects": redirects, "wikidata_id_mapping": wikidata_id_mapping},
        remove_columns=["html"],
        batched=True,
    )
    dataset = dataset.filter(
        lambda x: x["text"] != "" and len(x["anchors"]) > 0,
    )
    dataset = datasets.Dataset.from_generator(lambda: (yield from dataset))
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikiextractor_output_dir", type=str, required=True)
    parser.add_argument("--redirect_file", type=str, required=True)
    parser.add_argument("--wikidata_id_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
