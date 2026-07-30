"""
Module 5 — Advantage Calculator.

Compute group-relative advantage per the GRPO formula:
  A_i = (R_i - mean) / (std + eps)
"""

import math


def compute_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    if not rewards:
        return []

    n = len(rewards)
    mean = sum(rewards) / n
    # population variance/std (divide by N, not N-1)
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    denom = std + eps  # eps guards against division by zero when std == 0
    return [(r - mean) / denom for r in rewards]


def flag_zero_variance_group(rewards: list[float]) -> bool:
    if len(rewards) <= 1:
        return True
    return max(rewards) == min(rewards)
