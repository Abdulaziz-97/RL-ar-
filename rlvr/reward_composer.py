"""
Module 3 — Reward Composer.

Composite reward: 0.6 * correctness + 0.2 * format + 0.05 * language
minus answer-leak and structural-leak penalties, clamped to [0.0, 1.0].
"""

import json
import math
import re
from typing import Optional, Callable

W_CORRECTNESS = 0.6
W_FORMAT = 0.2
W_LANGUAGE = 0.05
W_ANSWER_LEAK = 0.5
W_STRUCTURAL_LEAK = 0.3

DEFAULT_LEAK_PHRASES = [
    "the answer is definitely",
    "the answer is",
    "the final answer is",
    "so the answer is",
    "الإجابة النهائية هي",
    "الإجابة هي",
    "الجواب هو",
    "الإجابة النهائية",
]

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_DIGIT_RE = re.compile(r"^[\d\u0660-\u0669.\-+*/=<>()]+$")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_think(completion: str) -> Optional[str]:
    m = _THINK_RE.search(completion)
    return m.group(1) if m else None


def _extract_answer(completion: str) -> Optional[str]:
    m = _ANSWER_RE.search(completion)
    return m.group(1).strip() if m else None


def _try_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reward_correctness(completion: str, ground_truth, domain: str, puzzle_type: Optional[str] = None) -> float:
    # Gate on structural validity first — prevents reward-hacking by emitting
    # bare <answer> tags without any <think> reasoning block.
    if "<think>" not in completion or "</think>" not in completion:
        return 0.0
    if "<answer>" not in completion or "</answer>" not in completion:
        return 0.0

    raw = _extract_answer(completion)
    if raw is None:
        return 0.0

    if domain == "math":
        g_num = _try_float(ground_truth)
        a_num = _try_float(raw)
        if g_num is not None and a_num is not None:
            return 1.0 if abs(g_num - a_num) < 1e-6 else 0.0
        return 1.0 if str(raw).strip().lower() == str(ground_truth).strip().lower() else 0.0

    if domain == "logic":
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = raw
        if isinstance(ground_truth, dict) and isinstance(parsed, dict):
            return 1.0 if parsed == ground_truth else 0.0
        if isinstance(ground_truth, (list, tuple, set)):
            try:
                return 1.0 if set(parsed) == set(ground_truth) else 0.0
            except TypeError:
                return 0.0
        return 1.0 if parsed == ground_truth else 0.0

    return 0.0


def reward_format(completion: str) -> float:
    if "<think>" not in completion or "</think>" not in completion:
        return 0.0
    if "<answer>" not in completion or "</answer>" not in completion:
        return 0.0

    think_text = _extract_think(completion)
    if think_text is None or think_text.strip() == "":
        return 0.0

    answer_text = _extract_answer(completion)
    if answer_text is None or answer_text == "":
        return 0.0

    tokens = think_text.split()
    total = len(tokens)
    if total == 0:
        return 0.0

    unique_ratio = len(set(tokens)) / total
    if unique_ratio < 0.4:
        return 0.0

    score = 1.0
    if total < 10:
        score -= 0.3
    if unique_ratio < 0.6:
        score -= 0.2
    return max(0.0, min(1.0, score))


def reward_language_consistency(completion: str, target_lang: str = "ar") -> float:
    if target_lang != "ar":
        return 1.0
    think_text = _extract_think(completion)
    if think_text is None:
        think_text = completion

    if think_text.strip() == "":
        return 0.0

    tokens = think_text.split()
    word_tokens = [t for t in tokens if not _DIGIT_RE.match(t)]
    if len(word_tokens) == 0:
        return 1.0

    arabic_count = sum(1 for t in word_tokens if _ARABIC_RE.search(t))
    return max(0.0, min(1.0, arabic_count / len(word_tokens)))


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def penalty_answer_leak(completion: str, embed_fn, leak_phrase_bank: list[str], threshold: float = 0.75) -> float:
    think_text = _extract_think(completion)
    if think_text is None or think_text.strip() == "":
        return 0.0

    if embed_fn is None:
        lowered = think_text.lower()
        for phrase in leak_phrase_bank:
            if phrase.lower() in lowered:
                return 1.0
        return 0.0

    think_vec = embed_fn(think_text)
    for phrase in leak_phrase_bank:
        sim = _cosine_similarity(think_vec, embed_fn(phrase))
        if sim > threshold:
            return 1.0
    return 0.0


def penalty_structural_leak(completion: str, max_preamble_words: int = 5) -> float:
    think_pos = completion.find("<think>")
    if think_pos == -1:
        # Missing <think> entirely is itself a structural violation.
        return 1.0

    answer_end_pos = completion.rfind("</answer>")
    preamble = completion[:think_pos]
    postamble = completion[answer_end_pos + len("</answer>"):] if answer_end_pos != -1 else ""

    outside = (preamble + " " + postamble).split()
    if len(outside) > max_preamble_words:
        return 1.0
    return 0.0


def compose_reward(completion: str, ground_truth, domain: str, puzzle_type: Optional[str] = None) -> float:
    r_correct = reward_correctness(completion, ground_truth, domain, puzzle_type)
    r_format = reward_format(completion)
    r_lang = reward_language_consistency(completion)
    p_leak = penalty_answer_leak(completion, None, DEFAULT_LEAK_PHRASES)
    p_struct = penalty_structural_leak(completion)

    total = (
        W_CORRECTNESS * r_correct
        + W_FORMAT * r_format
        + W_LANGUAGE * r_lang
        - W_ANSWER_LEAK * p_leak
        - W_STRUCTURAL_LEAK * p_struct
    )
    return max(0.0, min(1.0, total))
