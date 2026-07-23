"""
Configuration for AraEval evaluation suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


ARAEVAL_DATASETS: dict[str, str] = {
    "ara_ifeval": "humain-ai/AraIFEval",
    "ara_truthfulqa": "humain-ai/AraTruthfulQA",
    "ara_math": "humain-ai/AraMath",
    "ien_mcq": "humain-ai/IEN_MCQ",
    "ien_tf": "humain-ai/IEN_TF",
    "etec": "humain-ai/Etec",
    "lc_eval": "humain-ai/LC-Eval",
}


@dataclass
class AraEvalConfig:
    model_name_or_path: str = "Qwen/Qwen3.5-4B"
    adapter_path: Optional[str] = None
    tasks: list[str] = field(default_factory=lambda: ["all"])
    eval_mode: Literal["generation", "loglik"] = "generation"
    batch_size: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    limit: Optional[int] = None
    output_dir: str = "./araeval_results"
    load_in_4bit: bool = False
    bf16: bool = True
    fp16: bool = False
    device: str = "cuda"

    def get_task_list(self) -> list[str]:
        if "all" in self.tasks or not self.tasks:
            return list(ARAEVAL_DATASETS.keys())
        resolved = []
        raw_items = []
        for t in self.tasks:
            raw_items.extend([item.strip() for item in t.split(",") if item.strip()])

        for key in raw_items:
            key_clean = key.lower()
            if key_clean in ARAEVAL_DATASETS:
                resolved.append(key_clean)
            elif f"ara_{key_clean}" in ARAEVAL_DATASETS:
                resolved.append(f"ara_{key_clean}")
            else:
                match = next((k for k in ARAEVAL_DATASETS if key_clean in k), None)
                if match:
                    resolved.append(match)
        resolved = list(dict.fromkeys(resolved))
        return resolved or list(ARAEVAL_DATASETS.keys())
