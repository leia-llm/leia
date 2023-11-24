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
