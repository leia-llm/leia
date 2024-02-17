import argparse
import bz2
import re
from xml.etree.ElementTree import iterparse

from leia.utils import normalize_wikipedia_title


def _extract_wikipedia_redirects(dump_file: str) -> dict[str, str]:
    redirects = {}
    with bz2.open(dump_file, "rt") as f:
        elems = (elem for (_, elem) in iterparse(f, events=("end",)))
        head_elem = next(elems)
        tag = head_elem.tag
        namespace = _get_namespace(tag)
        page_tag = f"{{{namespace}}}page"
        title_path = f"./{{{namespace}}}title"
        redirect_path = f"./{{{namespace}}}redirect"

        for elem in elems:
            if elem.tag == page_tag:
                title = elem.find(title_path).text
                redirect = elem.find(redirect_path)
                if redirect is not None:
                    redirect = normalize_wikipedia_title(redirect.attrib["title"])
                    redirects[title] = redirect
                elem.clear()

    return redirects


def _get_namespace(tag: str) -> str:
    match_obj = re.match(r"^{(.*?)}", tag)
    if match_obj:
        namespace = match_obj.group(1)
        if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
            raise ValueError(f"{namespace} is not a valid MediaWiki dump namespace")
        return namespace
    else:
        return ""


def main(args: argparse.Namespace) -> None:
    redirects = _extract_wikipedia_redirects(args.dump_file)
    circular_redirects = set()

    for title in list(redirects.keys()):
        history = [title]
        while True:
            title = redirects[title]
            if title in circular_redirects:
                circular_redirects.update(history)
                break
            if title in history:
                circular_redirects.update(history)
                break

            if title in redirects:
                history.append(title)
            else:
                for src_title in history:
                    redirects[src_title] = title
                break

    with open(args.output_file, "w") as output_file:
        for title, redirect in redirects.items():
            if title not in circular_redirects:
                output_file.write(f"{title}\t{redirect}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
