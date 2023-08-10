from .base import BaseTask, GenerationTask, LoglikelihoodTask
from .jcommonsenseqa import JCommonsenseQA
from .jnli import JNLI
from .jsquad import JSQuAD
from .marc_ja import MARCJa
from .miaqa import MKQAEn, MKQAJa, XORQAEn, XORQAJa

_TASKS = {
    "jcommonsenseqa": JCommonsenseQA,
    "jnli": JNLI,
    "jsquad": JSQuAD,
    "marc_ja": MARCJa,
    "mkqa_en": MKQAEn,
    "mkqa_ja": MKQAJa,
    "xorqa_en": XORQAEn,
    "xorqa_ja": XORQAJa,
}


def get_task(task_name: str) -> BaseTask:
    return _TASKS[task_name]
