"""
Configuration for English baseline evaluation suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


ENGLISH_DATASETS: dict[str, str] = {
    "ifeval": "google/ifeval",
    "mmlu": "cais/mmlu",
    "hellaswag": "Rowan/hellaswag",
    "gsm8k": "gsm8k",
}

DEFAULT_ENGLISH_TASKS: list[str] = [
    "ifeval",
    "mmlu",
    "hellaswag",
    "gsm8k",
]

# Standard MMLU core subjects for quick evaluation
MMLU_CORE_SUBJECTS: list[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_computer_science",
    "college_mathematics",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_physics",
    "machine_learning",
    "philosophy",
    "professional_law",
]


@dataclass
class EnglishEvalConfig:
    model_name_or_path: str = "Qwen/Qwen3.5-4B"
    adapter_path: Optional[str] = None
    tasks: list[str] = field(default_factory=lambda: ["all"])
    mmlu_subjects: list[str] = field(default_factory=lambda: MMLU_CORE_SUBJECTS)
    batch_size: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    limit: Optional[int] = None
    output_dir: str = "./english_eval_results"
    load_in_4bit: bool = False
    bf16: bool = True
    fp16: bool = False
    device: str = "cuda"

    def get_task_list(self) -> list[str]:
        if "all" in self.tasks or not self.tasks:
            return DEFAULT_ENGLISH_TASKS
        resolved = []
        raw_items = []
        for t in self.tasks:
            raw_items.extend([item.strip() for item in t.split(",") if item.strip()])

        for key in raw_items:
            key_clean = key.lower()
            if key_clean in ENGLISH_DATASETS:
                resolved.append(key_clean)
            else:
                match = next((k for k in ENGLISH_DATASETS if key_clean in k), None)
                if match:
                    resolved.append(match)
        resolved = list(dict.fromkeys(resolved))
        return resolved or DEFAULT_ENGLISH_TASKS
