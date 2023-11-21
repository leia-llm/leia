import numpy as np
from datasets import Dataset, load_dataset

from .base import LogLikelihoodRequest, LoglikelihoodTask


class CmmluBase(LoglikelihoodTask):
    def _example_to_text(self, example: dict) -> str:
        return f"Question: {example['Question']}\nAnswer:"

    def _example_to_target(self, example: dict) -> str:
        return " " + example[example["Answer"]]

    @staticmethod
    def _get_answer_index(example: dict) -> int:
        return ["A", "B", "C", "D"].index(example["Answer"])

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


def _create_task_class(category: str) -> type[CmmluBase]:
    class _Cmmlu(CmmluBase):
        def _get_train_dataset(self) -> None:
            return load_dataset("haonan-li/cmmlu", category, split="dev")

        def _get_task_dataset(self) -> Dataset:
            return load_dataset("haonan-li/cmmlu", category, split="test")

    return _Cmmlu


def get_task_mapping() -> dict[str, type[CmmluBase]]:
    tasks = {}
    for category in [
        "agronomy",
        "anatomy",
        "ancient_chinese",
        "arts",
        "astronomy",
        "business_ethics",
        "chinese_civil_service_exam",
        "chinese_driving_rule",
        "chinese_food_culture",
        "chinese_foreign_policy",
        "chinese_history",
        "chinese_literature",
        "chinese_teacher_qualification",
        "clinical_knowledge",
        "college_actuarial_science",
        "college_education",
        "college_engineering_hydrology",
        "college_law",
        "college_mathematics",
        "college_medical_statistics",
        "college_medicine",
        "computer_science",
        "computer_security",
        "conceptual_physics",
        "construction_project_management",
        "economics",
        "education",
        "electrical_engineering",
        "elementary_chinese",
        "elementary_commonsense",
        "elementary_information_and_technology",
        "elementary_mathematics",
        "ethnology",
        "food_science",
        "genetics",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_geography",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "human_sexuality",
        "international_law",
        "journalism",
        "jurisprudence",
        "legal_and_moral_basis",
        "logical",
        "machine_learning",
        "management",
        "marketing",
        "marxist_theory",
        "modern_chinese",
        "nutrition",
        "philosophy",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_study",
        "sociology",
        "sports_science",
        "traditional_chinese_medicine",
        "virology",
        "world_history",
        "world_religions",
    ]:
        tasks[f"cmmlu_{category}"] = _create_task_class(category)

    return tasks
