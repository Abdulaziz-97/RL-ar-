"""
E2H Reasoner Gaussian Curriculum Scheduler.

At every training step, determine which difficulty bucket
(trivial/easy/medium/hard) to sample prompts from, using a Gaussian schedule
that smoothly transitions from easy to hard. This prevents task forgetting
(unlike a naive fixed-switch) by never assigning exactly zero probability to
any bucket.
"""

import math
import random
from typing import Literal

BUCKET_NAMES = ("trivial", "easy", "medium", "hard")

CurriculumScheduleType = Literal["gaussian", "fixed_switch", "random_mix", "none"]


def compute_stage_sampling_weights(
    step: int,
    total_steps: int,
    sigma_fraction: float = 0.2,
) -> dict[str, float]:
    """
    Returns a probability distribution over {"trivial", "easy", "medium", "hard"}
    for the given training step, using a Gaussian schedule centered on each
    bucket's target step.
    """
    if total_steps <= 0:
        return {name: 0.25 for name in BUCKET_NAMES}
    if not math.isfinite(sigma_fraction) or sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be finite and greater than zero")

    centers = {
        "trivial": 0.10 * total_steps,
        "easy": 0.35 * total_steps,
        "medium": 0.65 * total_steps,
        "hard": 0.90 * total_steps,
    }
    sigma = sigma_fraction * total_steps

    log_weights = {
        name: -((step - mu) ** 2) / (2.0 * sigma * sigma)
        for name, mu in centers.items()
    }
    max_log_weight = max(log_weights.values())
    raw_weights = {
        name: math.exp(log_weight - max_log_weight)
        for name, log_weight in log_weights.items()
    }

    total = sum(raw_weights.values())
    return {name: w / total for name, w in raw_weights.items()}


def fixed_switch_weights(step: int, total_steps: int) -> dict[str, float]:
    """Naive baseline: hard switch between buckets at fixed thresholds. Known to cause task forgetting."""
    if total_steps <= 0:
        return {name: 0.25 for name in BUCKET_NAMES}

    weights = {name: 0.0 for name in BUCKET_NAMES}
    if step < 0.25 * total_steps:
        weights["trivial"] = 1.0
    elif step < 0.50 * total_steps:
        weights["easy"] = 1.0
    elif step < 0.75 * total_steps:
        weights["medium"] = 1.0
    else:
        weights["hard"] = 1.0
    return weights


def random_mix_weights() -> dict[str, float]:
    """Naive baseline: uniform sampling across all buckets (no curriculum)."""
    return {name: 0.25 for name in BUCKET_NAMES}


def no_curriculum_weights() -> dict[str, float]:
    """Control condition: returns empty distribution signaling no curriculum filtering."""
    return {}


def get_curriculum_weights(
    schedule_type: CurriculumScheduleType,
    step: int,
    total_steps: int,
    sigma_fraction: float = 0.2,
) -> dict[str, float]:
    """Dispatch to the correct scheduler based on schedule_type flag."""
    if schedule_type == "gaussian":
        return compute_stage_sampling_weights(step, total_steps, sigma_fraction)
    if schedule_type == "fixed_switch":
        return fixed_switch_weights(step, total_steps)
    if schedule_type == "random_mix":
        return random_mix_weights()
    if schedule_type == "none":
        return no_curriculum_weights()
    raise ValueError(f"unknown schedule type: {schedule_type}")


def sample_curriculum_bucket(
    weights: dict[str, float],
    rng: random.Random | None = None,
) -> str | None:
    """Sample one bucket name from the weight distribution. Returns None if weights is empty (no curriculum)."""
    if not weights:
        return None

    draw = rng.random() if rng is not None else random.random()
    names = list(weights.keys())
    cumulative = 0.0
    for name in names:
        cumulative += weights[name]
        if draw < cumulative:
            return name
    return names[-1]
