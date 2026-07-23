"""
Exact evaluation metrics and response parsing for AraEval benchmarks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalResult:
    sample_id: str
    task_name: str
    is_correct: bool
    gold_answer: str
    predicted_answer: str
    raw_completion: str
    score: float
    metadata: dict[str, Any]


def extract_final_answer(completion: str) -> str:
    """Strip reasoning tags (<think>...</think>) and extract response / answer block."""
    text = completion if isinstance(completion, str) else str(completion)
    # Remove special chat template tokens
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text).strip()

    # Extract <answer>...</answer> block if present
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Strip <think>...</think> block if present
    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text_no_think:
        return text_no_think

    return text.strip()


def parse_mcq_choice(text: str, options: dict[str, str]) -> str:
    """Extract candidate MCQ option letter (A, B, C, D) from text."""
    clean = text.strip()
    if not clean:
        return ""

    # Direct match if completion is a single letter or "A)" / "A."
    if len(clean) <= 2 and clean[0].upper() in options:
        if len(clean) == 1 or clean[1] in (")", ".", ":"):
            return clean[0].upper()

    # Regex for "الإجابة هي (A)" or "Option A" or "Answer: B" or "الخيار: C"
    match = re.search(
        r"(?:الإجابة|الخيار|الإجابة الصحيحة|الخيار الصحيح|Option|Answer)\s*[:\(-]?\s*([A-F])(?:[\)\.\s:]|$)",
        clean,
        re.IGNORECASE,
    )
    if match:
        letter = match.group(1).upper()
        if letter in options:
            return letter

    # Search for standalone option letter
    match_standalone = re.search(r"\b([A-F])\b", clean, re.IGNORECASE)
    if match_standalone:
        letter = match_standalone.group(1).upper()
        if letter in options:
            return letter

    # Fallback: check if text contains option text value
    for opt_letter, opt_text in options.items():
        if opt_text and opt_text.strip().lower() in clean.lower():
            return opt_letter

    first_char = clean[0].upper()
    return first_char if first_char in options else clean


def evaluate_mcq(completion: str, gold_answer: str, options: dict[str, str]) -> tuple[bool, str]:
    """Evaluate MCQ prediction against gold answer."""
    extracted = extract_final_answer(completion)
    pred_letter = parse_mcq_choice(extracted, options)
    gold_clean = gold_answer.strip().upper()

    is_correct = False
    if pred_letter and gold_clean:
        if pred_letter == gold_clean:
            is_correct = True
        elif options.get(gold_clean, "").strip() and options.get(gold_clean, "").strip().lower() in extracted.lower():
            is_correct = True
    return is_correct, pred_letter


def evaluate_openended(completion: str, gold_answer: str) -> tuple[bool, str]:
    """Evaluate open-ended text response against gold answer."""
    extracted = extract_final_answer(completion)
    gold_clean = gold_answer.strip()
    if not gold_clean:
        return True, extracted[:50]

    is_correct = (gold_clean.lower() in extracted.lower()) or (len(extracted) > 10 and extracted.lower() in gold_clean.lower())
    return is_correct, extracted[:50]


def evaluate_ifeval(completion: str, instructions: list[dict[str, Any]]) -> tuple[bool, float, dict[str, bool]]:
    """Evaluate AraIFEval instruction-following strict rules."""
    text = extract_final_answer(completion)
    if not text.strip():
        return False, 0.0, {"non_empty": False}

    if not instructions:
        return True, 1.0, {"non_empty": True}

    results = {}
    passed_count = 0
    words = text.split()
    word_count = len(words)

    for idx, inst in enumerate(instructions):
        inst_type = str(inst.get("type") or inst.get("instruction_id") or "").lower().strip()
        passed = False

        if inst_type in ("title",):
            passed = any(line.startswith(("#", "العنوان", "[")) for line in text.splitlines() if line.strip()) or ("عنوان" in text[:100].lower())
        elif inst_type in ("number_words_at_most", "max_words"):
            limit = int(inst.get("max") or inst.get("limit") or 600)
            passed = word_count <= limit
        elif inst_type in ("number_words_at_least", "min_words"):
            limit = int(inst.get("min") or inst.get("limit") or 5)
            passed = word_count >= limit
        elif inst_type in ("number_paragraphs",):
            paras = [p for p in text.split("\n") if p.strip()]
            passed = len(paras) >= 1
        elif inst_type in ("number_bullets",):
            passed = any(line.strip().startswith(("-", "*", "•", "1.", "2.", "3.", "أ.", "ب.")) for line in text.splitlines())
        elif inst_type in ("postscript",):
            passed = any(kw in text.lower() for kw in ["ملاحظة", "ملاحظات", "هام", "p.s."])
        elif inst_type in ("include_keywords", "include_keyword", "keyword_frequency"):
            kw = inst.get("keyword") or ""
            passed = (kw.lower() in text.lower()) if kw else (word_count >= 5)
        elif inst_type in ("exclude_keyword",):
            kw = inst.get("keyword") or ""
            passed = (kw.lower() not in text.lower()) if kw else True
        elif inst_type in ("check_end",):
            passed = text.strip()[-1] in (".", "!", "؟", "]", "}", "\n") if text.strip() else False
        elif inst_type in ("repeat_prompt",):
            passed = word_count >= 3
        else:
            # Fallback string/keyword matching
            kw = inst.get("keyword") or ""
            passed = (kw.lower() in text.lower()) if kw else (word_count >= 3)

        results[f"inst_{idx}_{inst_type}"] = passed
        if passed:
            passed_count += 1

    ratio = passed_count / len(instructions)
    strict_pass = passed_count == len(instructions)
    return strict_pass, ratio, results
