"""
Module 3 — Reward Composer.

Composite reward: 0.6 * correctness + 0.2 * format + 0.2 * language
minus answer-leak, structural-leak, and length penalties, clamped to [0.0, 1.0].

Format/correctness/leak now delegate to ``rlvr_contracts`` so generation
acceptance and training share identical behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional

# Prefer local pack layout: src/rlvr + src/rlvr_contracts
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from rlvr_contracts.answer_spec import AnswerSpec, AnswerSpecError, parse_answer_spec, reject_symbolic
from rlvr_contracts.leak import (
    DEFAULT_LEAK_PHRASES,
    leak_policy_score,
    length_penalty_score,
    structural_leak_score,
)
from rlvr_contracts.response import (
    arabic_language_score,
    extract_answer,
    extract_think,
    validate_response_format,
)
from rlvr_contracts.verifiers import verify_answer, verify_legacy

W_CORRECTNESS = 0.6
W_FORMAT = 0.2
W_LANGUAGE = 0.2
W_ANSWER_LEAK = 0.5
W_STRUCTURAL_LEAK = 0.3
W_LENGTH = 0.15

# Token-approx length policy (aligned with 640–768 completion capacity).
THINK_LENGTH_SOFT_LIMIT = 256
THINK_LENGTH_HARD_LIMIT = 512

# Back-compat aliases used by older tests / callers.
_extract_think = extract_think
_extract_answer = extract_answer


def reward_correctness(
    completion: str,
    ground_truth,
    domain: str,
    puzzle_type: Optional[str] = None,
    answer_spec: Optional[Mapping[str, Any] | AnswerSpec] = None,
) -> float:
    """Score correctness via shared verifiers.

    Prefer ``answer_spec`` when present. Symbolic specs are rejected (score 0).
    """
    if answer_spec is not None:
        try:
            spec = parse_answer_spec(answer_spec)
            reject_symbolic(spec.type)
        except AnswerSpecError:
            return 0.0
        return float(verify_answer(completion, spec, from_completion=True).score)

    return float(verify_legacy(completion, ground_truth, domain, puzzle_type))


def reward_format(completion: str) -> float:
    return float(validate_response_format(completion))


def reward_language_consistency(completion: str, target_lang: str = "ar") -> float:
    return float(arabic_language_score(completion, target_lang=target_lang))


def penalty_answer_leak(completion: str, embed_fn, leak_phrase_bank: list[str], threshold: float = 0.75) -> float:
    return float(
        leak_policy_score(
            completion,
            leak_phrase_bank=leak_phrase_bank,
            embed_fn=embed_fn,
            threshold=threshold,
        )
    )


def penalty_structural_leak(completion: str, max_preamble_words: int = 5) -> float:
    return float(structural_leak_score(completion, max_preamble_words=max_preamble_words))


def penalty_length(
    completion: str,
    soft_limit: int = THINK_LENGTH_SOFT_LIMIT,
    hard_limit: int = THINK_LENGTH_HARD_LIMIT,
) -> float:
    return float(length_penalty_score(completion, soft_limit=soft_limit, hard_limit=hard_limit))


def compose_reward(
    completion: str,
    ground_truth,
    domain: str,
    puzzle_type: Optional[str] = None,
    answer_spec: Optional[Mapping[str, Any] | AnswerSpec] = None,
) -> float:
    r_correct = reward_correctness(
        completion, ground_truth, domain, puzzle_type, answer_spec=answer_spec
    )
    r_format = reward_format(completion)
    r_lang = reward_language_consistency(completion)
    p_leak = penalty_answer_leak(completion, None, DEFAULT_LEAK_PHRASES)
    p_struct = penalty_structural_leak(completion)
    p_length = penalty_length(completion)

    total = (
        W_CORRECTNESS * r_correct
        + W_FORMAT * r_format
        + W_LANGUAGE * r_lang
        - W_ANSWER_LEAK * p_leak
        - W_STRUCTURAL_LEAK * p_struct
        - W_LENGTH * p_length
    )
    return max(0.0, min(1.0, total))
