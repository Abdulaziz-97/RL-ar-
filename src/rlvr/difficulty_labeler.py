"""
Module 4 — Difficulty Labeler (post-hoc refinement).

Refine coarse easy/hard labels into empirical error-rate quartiles
(trivial/easy/medium/hard), computed relative to the actual base model.
"""

import hashlib
from typing import Protocol, runtime_checkable

from rlvr.reward_composer import reward_correctness


@runtime_checkable
class GenerativeModel(Protocol):
    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str: ...


QUARTILE_LABELS = ("trivial", "easy", "medium", "hard")


def compute_error_rate(model, sample, n_attempts: int = 20) -> float:
    prompt = sample["prompt"]
    ground_truth = sample["ground_truth_answer"]
    domain = sample["domain"]
    puzzle_type = sample.get("puzzle_type")

    correct = 0
    for _ in range(n_attempts):
        completion = model.generate(prompt, temperature=0.7, max_new_tokens=512)
        if reward_correctness(completion, ground_truth, domain, puzzle_type) >= 1.0:
            correct += 1
    return 1.0 - (correct / n_attempts)


def assign_difficulty_quartiles(samples, error_rates: list[float]):
    if len(samples) != len(error_rates):
        raise ValueError("samples and error_rates must have equal length")

    n = len(samples)
    if n == 0:
        return []

    indexed = sorted(range(n), key=lambda i: error_rates[i])
    refined = [dict(samples[i]) for i in range(n)]

    for rank, idx in enumerate(indexed):
        quartile = min(3, rank * 4 // n) if n > 0 else 0
        refined[idx]["difficulty_tag"] = QUARTILE_LABELS[quartile]

    return refined
