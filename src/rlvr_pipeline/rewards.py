"""
TRL-compatible reward functions for the Arabic Reasoning RLVR pipeline.

Each function follows the TRL signature:
    def reward_func(prompts, completions, **kwargs) -> list[float]

TRL sums weighted reward functions. With `reward_weights=[0.6, 0.2, 0.2, 0.5, 0.3]`
the total matches the PRD composite formula:
    0.6*correctness + 0.2*format + 0.2*language - 0.5*answer_leak - 0.3*structural_leak

Using separate functions (rather than one composite) lets TRL log each component
independently in the dashboard — the 2026 monitoring approach.
"""

from __future__ import annotations

from typing import Any

from rlvr.reward_composer import (
    DEFAULT_LEAK_PHRASES,
    W_ANSWER_LEAK,
    W_CORRECTNESS,
    W_FORMAT,
    W_LANGUAGE,
    W_STRUCTURAL_LEAK,
    compose_reward,
    penalty_answer_leak,
    penalty_structural_leak,
    reward_correctness,
    reward_format,
    reward_language_consistency,
)

REWARD_FUNCS_ORDER = ("correctness", "format", "language", "answer_leak", "structural_leak")


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        if len(completion) > 0 and isinstance(completion[0], str):
            return completion[0]
        return ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _extract_column(values: list[Any]) -> list[Any]:
    result = []
    for v in values:
        if isinstance(v, list) and len(v) > 0:
            result.append(v[0])
        else:
            result.append(v)
    return result


def correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    ground_truth = _extract_column(kwargs.get("ground_truth_answer", []))
    domain = _extract_column(kwargs.get("domain", []))
    puzzle_type = kwargs.get("puzzle_type", [None] * len(completions))
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        gt = ground_truth[i] if i < len(ground_truth) else None
        dom = domain[i] if i < len(domain) else "math"
        pt = puzzle_type[i] if i < len(puzzle_type) else None
        rewards.append(float(reward_correctness(text, gt, dom, pt)))
    return rewards


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    return [float(reward_format(_extract_completion_text(c))) for c in completions]


def language_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for c in completions:
        text = _extract_completion_text(c)
        score = float(reward_language_consistency(text))
        if "</answer>" not in text:
            score = 0.0
        rewards.append(score)
    return rewards


def answer_leak_penalty_func(prompts, completions, **kwargs) -> list[float]:
    return [
        -float(penalty_answer_leak(_extract_completion_text(c), None, DEFAULT_LEAK_PHRASES))
        for c in completions
    ]


def structural_leak_penalty_func(prompts, completions, **kwargs) -> list[float]:
    return [-float(penalty_structural_leak(_extract_completion_text(c))) for c in completions]


ALL_REWARD_FUNCS = [
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    answer_leak_penalty_func,
    structural_leak_penalty_func,
]

DEFAULT_REWARD_WEIGHTS = [W_CORRECTNESS, W_FORMAT, 0.05, W_ANSWER_LEAK, W_STRUCTURAL_LEAK]


def composite_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Single composite reward (alternative to the multi-function approach).

    Returns the clamped [0, 1] composite from the PRD. Use this when you want
    a single reward signal matching the v1 PRD exactly, instead of the
    multi-component approach.
    """
    ground_truth = _extract_column(kwargs.get("ground_truth_answer", []))
    domain = _extract_column(kwargs.get("domain", []))
    puzzle_type = kwargs.get("puzzle_type", [None] * len(completions))
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        gt = ground_truth[i] if i < len(ground_truth) else None
        dom = domain[i] if i < len(domain) else "math"
        pt = puzzle_type[i] if i < len(puzzle_type) else None
        rewards.append(float(compose_reward(text, gt, dom, pt)))
    return rewards
