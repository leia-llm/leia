import re
import string
from collections import Counter

import emoji
import neologdn
from datasets import Dataset, load_dataset
from fugashi import Tagger

from .base import GenerationRequest, GenerationTask

_tagger = None


# The above functions are based on the following code:
# https://github.com/Stability-AI/lm-evaluation-harness/blob/82ca7dd6f0eed2ea4ca957e73bb9c2048a2e5555/lm_eval/jasquad/evaluate.py
def remove_punc(tokens: list[str]):
    exclude = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    exclude += string.punctuation
    exclude = [*exclude]
    return [tok for tok in tokens if tok not in exclude]


def normalize_answer(s: str):
    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_emoji(text: str):
        text = "".join(["" if emoji.is_emoji(c) else c for c in text])
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    return white_space_fix((neologdn.normalize(remove_emoji(s))))


def f1_score(prediction: str, ground_truth: str):
    global _tagger
    if _tagger is None:
        _tagger = Tagger("-Owakati")

    prediction_tokens = remove_punc(_tagger.parse(normalize_answer(prediction)).split())
    ground_truth_tokens = remove_punc(_tagger.parse(normalize_answer(ground_truth)).split())
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


class JSQuAD(GenerationTask):
    def _get_description(self) -> str:
        return "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。\n\n"

    def _get_train_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JSQuAD", split="train")

    def _get_task_dataset(self) -> Dataset:
        return load_dataset("shunk031/JGLUE", "JSQuAD", split="validation")

    def _example_to_text(self, example: dict) -> str:
        return f"文章:{example['context'].split('[SEP]')[-1].strip()}\n質問:{example['question']}\n回答:"

    def _example_to_target(self, example: dict) -> str:
        answer_list = example["answers"]["text"]
        answer = answer_list[0]
        return answer

    def _create_requests(self, example: dict, context: str) -> list[GenerationRequest]:
        max_generation_length = max(
            len(self._tokenizer.encode(answer, add_special_tokens=False)) for answer in example["answers"]["text"]
        )
        requests = [GenerationRequest(context, stop_sequences=["\n"], max_generation_length=max_generation_length)]
        return requests

    def _process_results(self, example: dict, results: list[str]) -> dict:
        generated_text = results[0]
        answers = example["answers"]["text"]
        ret = {
            "exact_match": max(exact_match_score(generated_text, answer) for answer in answers),
            "f1": max(f1_score(generated_text, answer) for answer in answers),
            "prediction": generated_text,
        }
        return ret
