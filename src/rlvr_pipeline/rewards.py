"""TRL reward functions for Arabic Reasoning RLVR.

Signature: reward_func(prompts, completions, **kwargs) -> list[float]
Separate funcs so TRL logs each component; weights applied via reward_weights.
"""

from __future__ import annotations

import json
from typing import Any

from rlvr.reward_composer import (
    DEFAULT_LEAK_PHRASES,
    W_LENGTH,
    compose_reward,
    penalty_answer_leak,
    penalty_length,
    penalty_structural_leak,
    reward_correctness,
    reward_format,
    reward_language_consistency,
)

REWARD_FUNCS_ORDER = (
    "correctness",
    "format",
    "language",
    "answer_leak",
    "structural_leak",
    "length",
)


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


def _coerce_logic_ground_truth(gt: Any) -> Any:
    """HF loader stringifies dict GTs; reward_correctness needs a dict for logic."""
    if isinstance(gt, dict):
        return gt
    if isinstance(gt, str):
        text = gt.strip()
        if not text:
            return gt
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return gt
        if isinstance(parsed, dict):
            return parsed
    return gt


def _coerce_answer_spec(raw: Any) -> Any:
    if raw is None or raw == "" or raw == {}:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    ground_truth = _extract_column(kwargs.get("ground_truth_answer", []))
    domain = _extract_column(kwargs.get("domain", []))
    puzzle_type = _extract_column(kwargs.get("puzzle_type", [None] * len(completions)))
    answer_specs = _extract_column(kwargs.get("answer_spec", [None] * len(completions)))
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        gt = ground_truth[i] if i < len(ground_truth) else None
        dom = domain[i] if i < len(domain) else "math"
        pt = puzzle_type[i] if i < len(puzzle_type) else None
        spec = _coerce_answer_spec(answer_specs[i] if i < len(answer_specs) else None)
        if dom == "logic":
            gt = _coerce_logic_ground_truth(gt)
        rewards.append(float(reward_correctness(text, gt, dom, pt, answer_spec=spec)))
    return rewards


def _partial_format_score(text: str) -> float:
    """Small format shaping when strict format is 0. Empty tags score 0; max 0.30."""
    score = 0.0

    if "<think>" in text and "</think>" in text:
        start = text.find("<think>") + len("<think>")
        end = text.find("</think>")
        body = text[start:end].strip() if end > start else ""
        if body:
            score += 0.08
            if len(body.split()) >= 8:
                score += 0.07

    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        body = text[start:end].strip() if end > start else ""
        if body:
            score += 0.08
            if len(body) >= 1:
                score += 0.05

    return min(0.30, score)


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Strict format when it passes; otherwise partial shaping only."""
    rewards = []
    for c in completions:
        text = _extract_completion_text(c)
        strict = float(reward_format(text))
        if strict > 0.0:
            rewards.append(strict)
        else:
            rewards.append(_partial_format_score(text))
    return rewards


def language_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Arabic consistency on think text (not gated on </answer>)."""
    return [
        float(reward_language_consistency(_extract_completion_text(c)))
        for c in completions
    ]


def answer_leak_penalty_func(prompts, completions, **kwargs) -> list[float]:
    return [
        -float(penalty_answer_leak(_extract_completion_text(c), None, DEFAULT_LEAK_PHRASES))
        for c in completions
    ]


def structural_leak_penalty_func(prompts, completions, **kwargs) -> list[float]:
    return [-float(penalty_structural_leak(_extract_completion_text(c))) for c in completions]


def length_penalty_func(prompts, completions, **kwargs) -> list[float]:
    """Penalize runaway <think> length (thinking-loop pressure)."""
    return [-float(penalty_length(_extract_completion_text(c))) for c in completions]


ALL_REWARD_FUNCS = [
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    answer_leak_penalty_func,
    structural_leak_penalty_func,
    length_penalty_func,
]

DEFAULT_REWARD_WEIGHTS = [0.6, 0.2, 0.05, 0.5, 0.3, W_LENGTH]


def composite_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Single clamped [0, 1] composite reward (optional alternative to multi-func)."""
    ground_truth = _extract_column(kwargs.get("ground_truth_answer", []))
    domain = _extract_column(kwargs.get("domain", []))
    puzzle_type = _extract_column(kwargs.get("puzzle_type", [None] * len(completions)))
    answer_specs = _extract_column(kwargs.get("answer_spec", [None] * len(completions)))
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        gt = ground_truth[i] if i < len(ground_truth) else None
        dom = domain[i] if i < len(domain) else "math"
        pt = puzzle_type[i] if i < len(puzzle_type) else None
        spec = _coerce_answer_spec(answer_specs[i] if i < len(answer_specs) else None)
        if dom == "logic":
            gt = _coerce_logic_ground_truth(gt)
        rewards.append(float(compose_reward(text, gt, dom, pt, answer_spec=spec)))
    return rewards
