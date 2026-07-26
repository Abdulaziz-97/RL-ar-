"""Ingestion module for standard English baseline benchmarks (GSM8K, Hendrycks MATH).

Downloads or loads English reasoning benchmarks, extracts final ground truth answers,
infers valid canonical answer_specs (numeric/rational/decimal), and exports to .jsonl
matching the RLVR pipeline production schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from rlvr_contracts.answer_spec import infer_answer_spec_from_legacy, AnswerSpecError

DEFAULT_ENGLISH_SYSTEM_PROMPT = (
    "Please solve the following problem step-by-step. "
    "Write your reasoning clearly, and put your final answer inside <answer>...</answer> tags."
)


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract final numerical answer from GSM8K ground truth text ('... #### 42')."""
    if not answer_text or "####" not in answer_text:
        return None
    raw_ans = answer_text.split("####")[-1].strip()
    # Remove commas and dollar signs from numbers like $1,200 -> 1200
    clean_ans = raw_ans.replace(",", "").replace("$", "").replace("%", "").strip()
    return clean_ans if clean_ans else None


def extract_math_answer(solution_text: str) -> Optional[str]:
    """Extract boxed answer from Hendrycks MATH solution text ('... \\boxed{42}')."""
    if not solution_text:
        return None
    match = re.search(r"\\boxed\{([^{}]+)\}", solution_text)
    if match:
        clean = match.group(1).strip().replace(",", "").replace("$", "")
        return clean
    return None


def format_english_benchmark_record(
    question: str,
    raw_answer: str,
    sample_id: str,
    dataset_name: str = "gsm8k",
    system_prompt: str = DEFAULT_ENGLISH_SYSTEM_PROMPT,
) -> Optional[dict[str, Any]]:
    """Format a single raw problem into production RLVR JSONL schema."""
    if dataset_name == "gsm8k":
        target_str = extract_gsm8k_answer(raw_answer)
        domain = "gsm8k"
    else:
        target_str = extract_math_answer(raw_answer)
        domain = "math"

    if not target_str:
        return None

    try:
        spec = infer_answer_spec_from_legacy(target_str, domain="math")
    except AnswerSpecError:
        # Skip items that cannot be parsed into canonical numeric answer specs
        return None

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]

    return {
        "sample_id": sample_id,
        "family_id": dataset_name,
        "partition": "test",
        "domain": domain,
        "language": "en",
        "prompt": prompt_messages,
        "ground_truth": spec.canonical,
        "answer_spec": spec.to_dict(),
        "difficulty_tag": "medium",
    }


def convert_gsm8k_dataset(
    dataset_items: list[dict[str, Any]],
    system_prompt: str = DEFAULT_ENGLISH_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Convert a list of raw GSM8K dict items ({'question': ..., 'answer': ...}) into RLVR schema."""
    converted = []
    for idx, item in enumerate(dataset_items):
        q = item.get("question") or item.get("problem") or ""
        a = item.get("answer") or item.get("solution") or ""
        rec = format_english_benchmark_record(
            question=q,
            raw_answer=a,
            sample_id=f"gsm8k_test_{idx}",
            dataset_name="gsm8k",
            system_prompt=system_prompt,
        )
        if rec:
            converted.append(rec)
    return converted


def save_benchmark_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save formatted records to a .jsonl file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def fetch_and_export_gsm8k(
    output_path: str | Path = "data/english_gsm8k_test.jsonl",
    split: str = "test",
    system_prompt: str = DEFAULT_ENGLISH_SYSTEM_PROMPT,
) -> Path:
    """Fetch GSM8K test split via HuggingFace datasets and export to JSONL."""
    from datasets import load_dataset

    print(f"Fetching HuggingFace GSM8K ({split} split)...", flush=True)
    hf_dataset = load_dataset("gsm8k", "main", split=split)
    items = [dict(row) for row in hf_dataset]
    converted = convert_gsm8k_dataset(items, system_prompt=system_prompt)
    out = save_benchmark_jsonl(converted, output_path)
    print(f"Exported {len(converted)} / {len(items)} GSM8K records to {out}", flush=True)
    return out


if __name__ == "__main__":
    fetch_and_export_gsm8k()
