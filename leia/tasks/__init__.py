from .base import BaseTask, GenerationTask, LoglikelihoodTask
from .belebele import get_task_mapping as belebele_get_task_mapping
from .ceval import get_task_mapping as ceval_get_task_mapping
from .cmmlu import get_task_mapping as cmmlu_get_task_mapping
from .exams import get_task_mapping as exams_get_task_mapping
from .headqa import get_task_mapping as headqa_get_task_mapping
from .jaqket import Jaqket
from .jcommonsenseqa import JCommonsenseQA
from .jsquad import JSQuAD
from .llm_jp_eval import get_task_mapping as llm_jp_eval_get_task_mapping
from .miaqa import get_task_mapping as miaqa_get_task_mapping
from .pawsx import get_task_mapping as pawsx_get_task_mapping
from .xcodah import get_task_mapping as xcodah_get_task_mapping
from .xcopa import get_task_mapping as xcopa_get_task_mapping
from .xcsqa import get_task_mapping as xcsqa_get_task_mapping
from .xnli import get_task_mapping as xnli_get_task_mapping
from .xstorycloze import get_task_mapping as xstorycloze_get_task_mapping
from .xwinograd import get_task_mapping as xwinograd_get_task_mapping

_TASKS: dict[str, type[BaseTask]] = {
    "jaqket": Jaqket,
    "jcommonsenseqa": JCommonsenseQA,
    "jsquad": JSQuAD,
}
_TASKS.update(belebele_get_task_mapping())
_TASKS.update(ceval_get_task_mapping())
_TASKS.update(cmmlu_get_task_mapping())
_TASKS.update(exams_get_task_mapping())
_TASKS.update(headqa_get_task_mapping())
_TASKS.update(llm_jp_eval_get_task_mapping())
_TASKS.update(miaqa_get_task_mapping())
_TASKS.update(pawsx_get_task_mapping())
_TASKS.update(xcodah_get_task_mapping())
_TASKS.update(xcopa_get_task_mapping())
_TASKS.update(xcsqa_get_task_mapping())
_TASKS.update(xnli_get_task_mapping())
_TASKS.update(xstorycloze_get_task_mapping())
_TASKS.update(xwinograd_get_task_mapping())


def get_task(task_name: str) -> type[BaseTask]:
    return _TASKS[task_name]
