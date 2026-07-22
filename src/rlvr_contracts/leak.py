"""Leak / structural policies shared by generation acceptance and rewards.

Allows a supported derived final arithmetic line inside <think>, while rejecting
unsupported copying, verifier/ground-truth references, filler, and classic
answer assertions.
"""

from __future__ import annotations

import re
from typing import Optional

from rlvr_contracts.response import extract_answer, extract_think

# Classic assertion-style leaks (still banned).
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

# Unsupported meta / verifier references.
UNSUPPORTED_COPY_PATTERNS = [
    re.compile(r"\bground[\s_-]?truth\b", re.I),
    re.compile(r"\bexpected answer\b", re.I),
    re.compile(r"\bverifier\b", re.I),
    re.compile(r"المتحقق|الحقيقة المرجعية|الإجابة المتوقعة"),
]

# Filler / bureaucracy that should not pad traces.
FILLER_PATTERNS = [
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi cannot\b", re.I),
    re.compile(r"بناءً على ما سبق أعلاه مباشرة"),
]

# Allowed final arithmetic derivation line, e.g. "42" or "= 42" or "الناتج = 15".
_FINAL_ARITH_LINE = re.compile(
    r"(?m)^\s*(?:الناتج|النتيجة|إذن|اذا|إذًا|إذا)?\s*=?\s*(-?\d+(?:\.\d+)?)\s*$"
)
_ARITH_EQ_LINE = re.compile(
    r"(?m)^\s*.{0,40}=\s*(-?\d+(?:\.\d+)?)\s*$"
)


def _is_allowed_derived_final_line(think: str, answer: Optional[str]) -> bool:
    """True when think ends with a short derived arithmetic line matching answer."""
    if not answer:
        return False
    ans = answer.strip()
    # Only allow for simple numeric answers.
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", ans):
        return False
    lines = [ln.strip() for ln in think.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1]
    m = _FINAL_ARITH_LINE.match(last) or _ARITH_EQ_LINE.match(last)
    if not m:
        return False
    return m.group(1).lstrip("0") == ans.lstrip("0") or m.group(1) == ans


def leak_policy_score(
    completion: str,
    *,
    leak_phrase_bank: Optional[list[str]] = None,
    embed_fn=None,
    threshold: float = 0.75,
) -> float:
    """Return 1.0 when a leak is detected, else 0.0.

    A supported derived final arithmetic line that matches <answer> is NOT a leak.
    """
    think = extract_think(completion)
    if think is None or think.strip() == "":
        return 0.0

    answer = extract_answer(completion)
    bank = leak_phrase_bank if leak_phrase_bank is not None else DEFAULT_LEAK_PHRASES
    lowered = think.lower()

    # Strip the allowed final arithmetic line before phrase scan so "= 42" alone
    # does not trip assertion heuristics when it matches the answer.
    scan_text = think
    if _is_allowed_derived_final_line(think, answer):
        lines = think.rstrip().splitlines()
        scan_text = "\n".join(lines[:-1])
        lowered = scan_text.lower()

    for phrase in bank:
        if phrase.lower() in lowered:
            return 1.0

    for pat in UNSUPPORTED_COPY_PATTERNS:
        if pat.search(scan_text):
            return 1.0
    for pat in FILLER_PATTERNS:
        if pat.search(scan_text):
            return 1.0

    if embed_fn is not None:
        import math

        def _cos(a, b) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        think_vec = embed_fn(scan_text)
        for phrase in bank:
            if _cos(think_vec, embed_fn(phrase)) > threshold:
                return 1.0
    return 0.0


def structural_leak_score(completion: str, max_preamble_words: int = 5) -> float:
    lowered = completion.lower()
    if any(
        lowered.count(tag) != 1
        for tag in ("<think>", "</think>", "<answer>", "</answer>")
    ):
        return 1.0
    think_pos = lowered.find("<think>")
    answer_end_pos = lowered.rfind("</answer>")
    preamble = completion[:think_pos] if think_pos != -1 else ""
    postamble = (
        completion[answer_end_pos + len("</answer>") :] if answer_end_pos != -1 else ""
    )
    outside = (preamble + " " + postamble).split()
    return 1.0 if len(outside) > max_preamble_words else 0.0


def length_penalty_score(
    completion: str,
    soft_limit: int = 80,
    hard_limit: int = 160,
    char_soft_limit: int = 2048,
    char_hard_limit: int = 4096,
) -> float:
    """Soft word-count pressure on ``<think>`` (80 soft / 160 hard)."""
    if hard_limit <= soft_limit:
        raise ValueError("hard_limit must be greater than soft_limit")
    think = extract_think(completion)
    if think is None:
        think = completion
    n = len(think.split())
    if n <= soft_limit:
        word_score = 0.0
    elif n >= hard_limit:
        word_score = 1.0
    else:
        word_score = (n - soft_limit) / (hard_limit - soft_limit)

    chars = len(think)
    if chars <= char_soft_limit:
        char_score = 0.0
    elif chars >= char_hard_limit:
        char_score = 1.0
    else:
        char_score = (chars - char_soft_limit) / (char_hard_limit - char_soft_limit)
    return max(word_score, char_score)
