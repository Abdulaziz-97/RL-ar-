"""
Module 7 — Entropy Collapse Prevention (Clip-Cov / KL-Cov).

Prevent entropy collapse by gating gradient updates on high-covariance tokens.
"""

import math
from typing import Literal


def compute_token_covariance(log_probs: list[float], advantages: list[float]) -> list[float]:
    n = len(log_probs)
    if n == 0:
        return []
    if len(advantages) != n:
        raise ValueError("log_probs and advantages must have the same length")
    if n == 1:
        return [0.0]
    mean_lp = sum(log_probs) / n
    mean_a = sum(advantages) / n
    return [(log_probs[i] - mean_lp) * (advantages[i] - mean_a) / n for i in range(n)]


def _top_k_indices(values: list[float], k: int) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return order[:k]


def clip_cov_mask(covariances: list[float], clip_ratio: float = 0.02) -> list[bool]:
    n = len(covariances)
    if n == 0:
        return []
    count = int(round(n * clip_ratio))
    if count < 1 and clip_ratio > 0:
        count = 1
    if count < 0:
        count = 0
    top = set(_top_k_indices(covariances, count))
    return [i in top for i in range(n)]


def kl_cov_penalty(covariances: list[float], kl_coef: float = 1.0, top_k_ratio: float = 0.02) -> list[float]:
    n = len(covariances)
    if n == 0:
        return []
    count_k = int(round(n * top_k_ratio))
    if count_k < 1 and top_k_ratio > 0:
        count_k = 1
    if count_k < 0:
        count_k = 0
    top = set(_top_k_indices(covariances, count_k))
    max_cov = max(covariances)
    penalties: list[float] = []
    for i in range(n):
        if i in top:
            if max_cov != 0:
                penalties.append(kl_coef * (covariances[i] / max_cov))
            else:
                penalties.append(kl_coef)
        else:
            penalties.append(0.0)
    return penalties


def apply_entropy_guard(loss_terms: list[float], mask_or_penalty: list, mode: Literal["clip_cov", "kl_cov"]) -> list[float]:
    if len(loss_terms) != len(mask_or_penalty):
        raise ValueError("loss_terms and mask_or_penalty must have the same length")
    if mode == "clip_cov":
        return [0.0 if mask_or_penalty[i] else loss_terms[i] for i in range(len(loss_terms))]
    if mode == "kl_cov":
        return [loss_terms[i] + mask_or_penalty[i] for i in range(len(loss_terms))]
    raise ValueError(f"unknown mode: {mode}")


def compute_batch_entropy(log_probs_per_token: list[list[float]]) -> float:
    if not log_probs_per_token:
        return 0.0
    entropies: list[float] = []
    for lps in log_probs_per_token:
        if not lps:
            continue
        total = 0.0
        for lp in lps:
            clp = lp if lp <= 0.0 else 0.0
            p = math.exp(clp)
            total -= p * clp
        entropies.append(total)
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)
