"""Exact response grammar: <think>\\n…\\n</think>\\n<answer>…</answer>."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Primary grammar (preferred): think block then answer block.
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_DIGIT_TOKEN_RE = re.compile(r"^[\d\u0660-\u0669.\-+*/=<>()\[\]{},:]+$")


@dataclass(frozen=True)
class ParsedResponse:
    think: Optional[str]
    answer: Optional[str]
    raw: str
    format_ok: bool
    format_score: float
    errors: tuple[str, ...]


def parse_response(completion: str) -> ParsedResponse:
    """Parse a model completion into think/answer spans.

    Format is valid only when both tags exist, think is non-empty, answer is
    non-empty, and uniqueness of think tokens is not pathologically low.
    """
    text = completion if isinstance(completion, str) else str(completion)
    errors: list[str] = []

    think_m = _THINK_RE.search(text)
    answer_m = _ANSWER_RE.search(text)

    if think_m is None:
        errors.append("missing_think_block")
    if answer_m is None:
        errors.append("missing_answer_block")

    think = think_m.group(1) if think_m else None
    answer = answer_m.group(1).strip() if answer_m else None

    if think is not None and think.strip() == "":
        errors.append("empty_think")
        think = ""
    if answer is not None and answer == "":
        errors.append("empty_answer")

    format_score = 0.0
    format_ok = False
    if think is not None and answer is not None and think.strip() and answer:
        tokens = think.split()
        total = len(tokens)
        if total == 0:
            errors.append("empty_think_tokens")
        else:
            unique_ratio = len(set(tokens)) / total
            if unique_ratio < 0.4:
                errors.append("think_repetition")
                format_score = 0.0
            else:
                format_score = 1.0
                if total < 10:
                    format_score -= 0.3
                if unique_ratio < 0.6:
                    format_score -= 0.2
                format_score = max(0.0, min(1.0, format_score))
                format_ok = format_score > 0.0
                # Prefer think-before-answer ordering but do not hard-fail if both present.
                if think_m and answer_m and think_m.start() > answer_m.start():
                    errors.append("answer_before_think")
                    format_score = max(0.0, format_score - 0.2)

    return ParsedResponse(
        think=think,
        answer=answer,
        raw=text,
        format_ok=format_ok,
        format_score=format_score,
        errors=tuple(errors),
    )


def validate_response_format(completion: str) -> float:
    """Return format reward in [0, 1] matching training/generation acceptance."""
    return parse_response(completion).format_score


def extract_think(completion: str) -> Optional[str]:
    return parse_response(completion).think


def extract_answer(completion: str) -> Optional[str]:
    return parse_response(completion).answer


def arabic_language_score(completion: str, target_lang: str = "ar") -> float:
    """Arabic consistency on think text (word tokens that are not pure digits/ops)."""
    if target_lang != "ar":
        return 1.0
    parsed = parse_response(completion)
    think_text = parsed.think if parsed.think is not None else completion
    if think_text.strip() == "":
        return 0.0
    arabic_re = re.compile(r"[\u0600-\u06FF]")
    tokens = think_text.split()
    word_tokens = [t for t in tokens if not _DIGIT_TOKEN_RE.match(t)]
    if not word_tokens:
        return 1.0
    arabic_count = sum(1 for t in word_tokens if arabic_re.search(t))
    return max(0.0, min(1.0, arabic_count / len(word_tokens)))
