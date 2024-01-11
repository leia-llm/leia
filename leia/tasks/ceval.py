import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class CevalBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return f"Question: {example['question']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example[example["answer"]]

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return ["A", "B", "C", "D"].index(example["answer"])

    def _create_requests(self, example: dict, context: str) -> list[LogLikelihoodRequest]:
        requests = [
            LogLikelihoodRequest(context=context, continuation=" " + example[choice]) for choice in ["A", "B", "C", "D"]
        ]
        return requests

    def _process_results(self, example: dict, results: list[float]) -> dict:
        prediction = int(np.argmax(results))
        if prediction == self._get_answer_index(example):
            accuracy = 1.0
        else:
            accuracy = 0.0

        return {"accuracy": accuracy, "prediction": prediction}


def _create_task_class(category: str) -> type[CevalBase]:
    class _Ceval(CevalBase):
        def _get_train_dataset(self) -> None:
            return load_dataset("ceval/ceval-exam", category, split="dev")

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("ceval/ceval-exam", category, split="val")

    return _Ceval


def get_task_mapping() -> dict[str, type[CevalBase]]:
    tasks = {}
    for category in [
        "computer_network",
        "operating_system",
        "computer_architecture",
        "college_programming",
        "college_physics",
        "college_chemistry",
        "advanced_mathematics",
        "probability_and_statistics",
        "discrete_mathematics",
        "electrical_engineer",
        "metrology_engineer",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "middle_school_mathematics",
        "middle_school_biology",
        "middle_school_physics",
        "middle_school_chemistry",
        "veterinary_medicine",
        "college_economics",
        "business_administration",
        "marxism",
        "mao_zedong_thought",
        "education_science",
        "teacher_qualification",
        "high_school_politics",
        "high_school_geography",
        "middle_school_politics",
        "middle_school_geography",
        "modern_chinese_history",
        "ideological_and_moral_cultivation",
        "logic",
        "law",
        "chinese_language_and_literature",
        "art_studies",
        "professional_tour_guide",
        "legal_professional",
        "high_school_chinese",
        "high_school_history",
        "middle_school_history",
        "civil_servant",
        "sports_science",
        "plant_protection",
        "basic_medicine",
        "clinical_medicine",
        "urban_and_rural_planner",
        "accountant",
        "fire_engineer",
        "environmental_impact_assessment_engineer",
        "tax_accountant",
        "physician",
    ]:
        tasks[f"ceval_{category}"] = _create_task_class(category)

    return tasks
