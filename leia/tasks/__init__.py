from .base import BaseTask, GenerationTask, LoglikelihoodTask
from .jcommonsenseqa import JCommonsenseQA
from .jsquad import JSQuAD
from .llm_jp_eval import get_task_mapping as llm_jp_eval_get_task_mapping
from .xcodah import get_task_mapping as xcodah_get_task_mapping
from .xcsqa import get_task_mapping as xcsqa_get_task_mapping
from .xnli import get_task_mapping as xnli_get_task_mapping

_TASKS: dict[str, type[BaseTask]] = {"jcommonsenseqa": JCommonsenseQA, "jsquad": JSQuAD}
_TASKS.update(llm_jp_eval_get_task_mapping())
_TASKS.update(xcodah_get_task_mapping())
_TASKS.update(xcsqa_get_task_mapping())
_TASKS.update(xnli_get_task_mapping())


def get_task(task_name: str) -> type[BaseTask]:
    return _TASKS[task_name]
