"""
AraEval: An Arabic Multi-Task Evaluation Suite for Large Language Models.

Paper: https://aclanthology.org/2025.emnlp-main.1679/
Datasets: https://huggingface.co/collections/humain-ai/araeval-datasets-687760e04b12a7afb429a4a0
"""

from __future__ import annotations

from araeval.config import AraEvalConfig
from araeval.runner import AraEvalRunner, evaluate_model

__all__ = ["AraEvalConfig", "AraEvalRunner", "evaluate_model"]
