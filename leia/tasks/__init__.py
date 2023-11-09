from .base import BaseTask, GenerationTask, LoglikelihoodTask
from .belebele import get_task_mapping as belebele_get_task_mapping
from .exams import get_task_mapping as exams_get_task_mapping
from .jcommonsenseqa import JCommonsenseQA
from .jnli import JNLI
from .jsquad import JSQuAD
from .marc_ja import MARCJa
from .miaqa import MKQAEn, MKQAJa, XORQAEn, XORQAJa
from .pawsx import get_task_mapping as pawsx_get_task_mapping
from .xcopa import get_task_mapping as xcopa_get_task_mapping
from .xnli import get_task_mapping as xnli_get_task_mapping
from .xstorycloze import get_task_mapping as xstorycloze_get_task_mapping
from .xwinograd import get_task_mapping as xwinograd_get_task_mapping

_TASKS: dict[str, type[BaseTask]] = {
    "jcommonsenseqa": JCommonsenseQA,
    "jnli": JNLI,
    "jsquad": JSQuAD,
    "marc_ja": MARCJa,
    "mkqa_en": MKQAEn,
    "mkqa_ja": MKQAJa,
    "xorqa_en": XORQAEn,
    "xorqa_ja": XORQAJa,
}
_TASKS.update(belebele_get_task_mapping())
_TASKS.update(exams_get_task_mapping())
_TASKS.update(pawsx_get_task_mapping())
_TASKS.update(xcopa_get_task_mapping())
_TASKS.update(xnli_get_task_mapping())
_TASKS.update(xstorycloze_get_task_mapping())
_TASKS.update(xwinograd_get_task_mapping())


def get_task(task_name: str) -> type[BaseTask]:
    return _TASKS[task_name]
