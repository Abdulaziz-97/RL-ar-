"""
Evaluation metrics and response parsers for English baseline benchmarks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EnglishEvalResult:
    sample_id: str
    task_name: str
    is_correct: bool
    gold_answer: str
    predicted_answer: str
    raw_completion: str
    score: float
    metadata: dict[str, Any]


def extract_final_answer(completion: str) -> str:
    """Strip reasoning tags (<think>...</think>) and extract answer block."""
    text = completion if isinstance(completion, str) else str(completion)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text).strip()

    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text_no_think:
        return text_no_think

    return text.strip()


def parse_mcq_choice(text: str, options: dict[str, str]) -> str:
    """Extract candidate MCQ option letter (A, B, C, D) from English text."""
    clean = text.strip()
    if not clean:
        return ""

    # Direct single-letter match
    if len(clean) <= 2 and clean[0].upper() in options:
        if len(clean) == 1 or clean[1] in (")", ".", ":", " "):
            return clean[0].upper()

    # Regex search for "Option A", "Answer: B", "The correct choice is (C)"
    match = re.search(
        r"(?:Option|Answer|Choice|The correct answer is)\s*[:\(-]?\s*([A-F])(?:[\)\.\s:]|$)",
        clean,
        re.IGNORECASE,
    )
    if match:
        letter = match.group(1).upper()
        if letter in options:
            return letter

    # Standalone letter search
    match_standalone = re.search(r"\b([A-F])\b", clean, re.IGNORECASE)
    if match_standalone:
        letter = match_standalone.group(1).upper()
        if letter in options:
            return letter

    # Match option text content
    for opt_letter, opt_text in options.items():
        if opt_text and opt_text.strip().lower() in clean.lower():
            return opt_letter

    first_char = clean[0].upper()
    return first_char if first_char in options else clean


def evaluate_mcq(completion: str, gold_answer: str, options: dict[str, str]) -> tuple[bool, str]:
    """Evaluate MCQ choice against gold answer."""
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


def extract_gsm8k_number(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K ground truth or completion."""
    clean = extract_final_answer(text)
    
    # Check for explicit #### <num> format first
    hash_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    # Check for boxed format \boxed{<num>}
    boxed_match = re.search(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}", clean)
    if boxed_match:
        return boxed_match.group(1).replace(",", "")

    # Fallback to last number in clean text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", clean)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def evaluate_gsm8k(completion: str, gold_answer: str) -> tuple[bool, str, str]:
    """Evaluate GSM8K numerical reasoning answer."""
    pred_num = extract_gsm8k_number(completion)
    gold_num = extract_gsm8k_number(gold_answer)

    is_correct = False
    if pred_num is not None and gold_num is not None:
        try:
            is_correct = abs(float(pred_num) - float(gold_num)) < 1e-4
        except ValueError:
            is_correct = (pred_num.strip() == gold_num.strip())

    return is_correct, pred_num or "", gold_num or ""


def evaluate_ifeval_item(completion: str, instruction_ids: list[str], kwargs_list: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """Evaluate IFEval English formatting constraints.
    
    Supports key rule types:
    - keyword: include_keyword, forbidden_words
    - length: number_words, number_paragraphs, number_sentences
    - format: json_format, title, bullet_list, capital_letters
    """
    text = extract_final_answer(completion)
    passed_rules = 0
    total_rules = max(len(instruction_ids), 1)
    details = {}

    for inst_id, kw in zip(instruction_ids, kwargs_list):
        rule_passed = True
        inst_type = inst_id.lower()

        # Keyword inclusion / forbidden words
        if "keyword" in inst_type or "include" in inst_type:
            keywords = kw.get("keywords") or kw.get("keyword_list") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            for target in keywords:
                if target.lower() not in text.lower():
                    rule_passed = False
                    break
        elif "forbidden" in inst_type:
            forbidden = kw.get("forbidden_words") or kw.get("words") or []
            for f_word in forbidden:
                if f_word.lower() in text.lower():
                    rule_passed = False
                    break
        # Paragraph / word count constraints
        elif "paragraph" in inst_type:
            num_paras = len([p for p in text.split("\n\n") if p.strip()])
            target_paras = kw.get("num_paragraphs") or kw.get("number_paragraphs")
            if target_paras and num_paras != target_paras:
                rule_passed = False
        elif "word" in inst_type or "length" in inst_type:
            words = text.split()
            target_words = kw.get("num_words") or kw.get("number_words")
            relation = kw.get("relation", "at least")
            if target_words:
                if relation == "at least" and len(words) < target_words:
                    rule_passed = False
                elif relation == "at most" and len(words) > target_words:
                    rule_passed = False
        # Structural format constraints
        elif "json" in inst_type:
            try:
                # Find JSON block or parse text
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                json.loads(json_str)
            except Exception:
                rule_passed = False
        elif "title" in inst_type:
            if not text.startswith("#") and not (text.split("\n")[0].isupper()):
                rule_passed = False
        elif "bullet" in inst_type:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            bullet_lines = [l for l in lines if l.startswith(("-", "*", "•", "1.", "2."))]
            if len(bullet_lines) == 0:
                rule_passed = False

        if rule_passed:
            passed_rules += 1
        details[inst_id] = rule_passed

    score = passed_rules / total_rules
    all_passed = (passed_rules == total_rules)
    return all_passed, score, details
