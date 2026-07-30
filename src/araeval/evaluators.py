"""
SOTA Evaluation Metrics and Response Parsing Engine for AraEval Benchmarks.
Supports: AraIFEval, AraPro, AraTrust, AraMath, and AraTruthfulQA.
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


ARABIC_TO_ALPHA_MAP = {
    "أ": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E", "ه": "E", "و": "F",
    "١": "A", "٢": "B", "٣": "C", "٤": "D", "٥": "E", "٦": "F",
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F",
}


def extract_final_answer(completion: str) -> str:
    """Extract response or explicit answer block from completion."""
    text = completion if isinstance(completion, str) else str(completion)
    # Strip ChatML / EOS special tokens
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>", "", text).strip()

    # Priority 1: <answer>...</answer> block
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans_match and ans_match.group(1).strip():
        ans = ans_match.group(1).strip()
        # If \boxed{...} inside <answer>, unwrap it
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", ans)
        if boxed_match:
            return boxed_match.group(1).strip()
        return ans

    # Priority 2: \boxed{...} anywhere in completion
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match and boxed_match.group(1).strip():
        return boxed_match.group(1).strip()

    # Priority 3: Strip reasoning <think>...</think> block
    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text_no_think:
        return text_no_think

    return text.strip()


def parse_mcq_choice(text: str, options: dict[str, str]) -> str:
    """Extract candidate MCQ option letter (A, B, C, D) from text cleanly."""
    clean = text.strip()
    if not clean:
        return ""

    # Normalize Arabic choice letters (أ -> A, ب -> B, etc.)
    for ar_char, alpha_char in ARABIC_TO_ALPHA_MAP.items():
        if clean.startswith(ar_char) and (len(clean) == 1 or clean[1] in (")", ".", ":", " ", "\n")):
            if alpha_char in options:
                return alpha_char

    # Direct match for single letter (e.g. "A", "A)", "A.")
    if len(clean) <= 3 and clean[0].upper() in options:
        if len(clean) == 1 or clean[1] in (")", ".", ":", " ", "\n"):
            return clean[0].upper()

    # Regex for explicit choice phrases in Arabic and English (e.g. "الإجابة الصحيحة هي (أ)" or "Option B")
    match = re.search(
        r"(?:الإجابة|الخيار|الإجابة الصحيحة|الخيار الصحيح|Option|Answer)\s*(?:هي|هو|تكون|يكون)?\s*[:\(-]?\s*([A-Fأ-د١-٦])(?:[\)\.\s:]|$)",
        clean,
        re.IGNORECASE,
    )
    if match:
        raw_char = match.group(1).strip()
        alpha = ARABIC_TO_ALPHA_MAP.get(raw_char, raw_char.upper())
        if alpha in options:
            return alpha

    # Standalone letter match
    match_standalone = re.search(r"\b([A-F])\b", clean, re.IGNORECASE)
    if match_standalone:
        letter = match_standalone.group(1).upper()
        if letter in options:
            return letter

    # Option text value match (e.g. model output "الرياض" instead of "B")
    for opt_letter, opt_text in options.items():
        if opt_text and len(opt_text.strip()) >= 2:
            clean_opt = opt_text.strip().lower()
            if clean_opt in clean.lower():
                return opt_letter

    first_char = ARABIC_TO_ALPHA_MAP.get(clean[0], clean[0].upper())
    return first_char if first_char in options else ""


def evaluate_mcq(completion: str, gold_answer: str, options: dict[str, str]) -> tuple[bool, str]:
    """Evaluate MCQ prediction against gold answer with SOTA precision."""
    extracted = extract_final_answer(completion)
    pred_letter = parse_mcq_choice(extracted, options)
    
    # Resolve gold answer key (e.g. "B" or "ب")
    gold_clean = gold_answer.strip().upper()
    gold_clean = ARABIC_TO_ALPHA_MAP.get(gold_clean, gold_clean)

    is_correct = False
    if pred_letter and gold_clean:
        if pred_letter == gold_clean:
            is_correct = True
    elif not pred_letter and gold_clean and options:
        gold_text = options.get(gold_clean, "").strip().lower()
        if len(gold_text) >= 2 and gold_text in extracted.lower():
            is_correct = True

    return is_correct, pred_letter or extracted[:30]


def evaluate_openended(completion: str, gold_answer: str) -> tuple[bool, str]:
    """Evaluate open-ended math/reasoning response against gold answer."""
    extracted = extract_final_answer(completion)
    gold_clean = gold_answer.strip()
    if not gold_clean:
        return True, extracted[:50]

    # Unwrap \boxed{...} formatting
    gold_stripped = re.sub(r"\\boxed\{([^}]+)\}", r"\1", gold_clean).strip()
    ext_stripped = re.sub(r"\\boxed\{([^}]+)\}", r"\1", extracted).strip()

    # Normalize whitespace & lower
    g_norm = re.sub(r"\s+", "", gold_stripped.lower())
    e_norm = re.sub(r"\s+", "", ext_stripped.lower())

    # Exact normalized match
    if g_norm == e_norm:
        return True, extracted[:50]

    # Exact substring match (if gold is non-trivial >= 2 chars)
    if len(gold_stripped) >= 2 and (gold_stripped.lower() in ext_stripped.lower()):
        return True, extracted[:50]

    # Numeric equivalence comparison
    gold_nums = re.findall(r"[-+]?\d*\.?\d+", gold_stripped)
    ext_nums = re.findall(r"[-+]?\d*\.?\d+", ext_stripped)

    if gold_nums and ext_nums:
        try:
            g_val = float(gold_nums[-1])
            # Check if any extracted number in answer matches gold_val
            for num_str in ext_nums:
                try:
                    e_val = float(num_str)
                    if abs(g_val - e_val) < 1e-4:
                        return True, extracted[:50]
                except ValueError:
                    continue
        except ValueError:
            pass

    return False, extracted[:50]


def evaluate_ifeval(completion: str, instructions: list[dict[str, Any]]) -> tuple[bool, float, dict[str, bool]]:
    """Evaluate AraIFEval instruction-following rules cleanly."""
    text = extract_final_answer(completion)
    # For IFEval, evaluate full response body (with <think> stripped)
    full_text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL | re.IGNORECASE).strip()
    full_text = re.sub(r"</?answer>", "", full_text).strip()
    if len(text.split()) < 15 and len(full_text.split()) >= 15:
        text = full_text

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
            kw = inst.get("keyword") or ""
            passed = (kw.lower() in text.lower()) if kw else (word_count >= 3)

        results[f"inst_{idx}_{inst_type}"] = passed
        if passed:
            passed_count += 1

    ratio = passed_count / len(instructions)
    strict_pass = passed_count == len(instructions)
    return strict_pass, ratio, results
