"""
Dataset loading and schema normalization for AraEval benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from araeval.config import ARAEVAL_DATASETS


@dataclass
class AraEvalSample:
    id: str
    task_name: str
    prompt: str
    question: str
    options: dict[str, str] = field(default_factory=dict)
    gold_answer: str = ""
    instructions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def format_mcq_prompt(question: str, options: dict[str, str]) -> str:
    """Format a multiple choice question into standard Arabic MCQ prompt."""
    prompt = f"السؤال: {question}\n\nالخيارات:\n"
    for label, text in options.items():
        prompt += f"{label}) {text}\n"
    prompt += "\nاختر حرف الإجابة الصحيحة فقط (A, B, C, D):\n"
    return prompt


def normalize_sample(raw: dict[str, Any], task_name: str, idx: int) -> AraEvalSample:
    """Normalize raw dataset records from Hugging Face datasets into AraEvalSample."""
    sample_id = str(raw.get("id") or raw.get("question_id") or raw.get("key") or f"{task_name}_{idx}")
    question = str(raw.get("question") or raw.get("prompt") or raw.get("input") or "").strip()

    options: dict[str, str] = {}
    raw_options = raw.get("options") or raw.get("choices")
    if isinstance(raw_options, dict):
        options = {str(k).upper(): str(v) for k, v in raw_options.items()}
    elif isinstance(raw_options, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        options = {labels[i]: str(opt) for i, opt in enumerate(raw_options) if i < len(labels)}
    elif "option_a" in raw or "A" in raw:
        for lbl in ["A", "B", "C", "D"]:
            val = raw.get(lbl) or raw.get(f"option_{lbl.lower()}")
            if val:
                options[lbl] = str(val)

    gold = str(raw.get("answer") or raw.get("target") or raw.get("gold") or "").strip()

    instructions = []
    if "instruction_list" in raw or "instructions" in raw:
        instructions = raw.get("instruction_list") or raw.get("instructions") or []

    if options and not question.startswith("السؤال:"):
        prompt = format_mcq_prompt(question, options)
    else:
        prompt = question or str(raw)

    return AraEvalSample(
        id=sample_id,
        task_name=task_name,
        prompt=prompt,
        question=question,
        options=options,
        gold_answer=gold,
        instructions=instructions,
        metadata=raw,
    )


def load_araeval_task(task_name: str, limit: Optional[int] = None) -> list[AraEvalSample]:
    """Load a specific AraEval task from Hugging Face or fallback mock data."""
    hf_path = ARAEVAL_DATASETS.get(task_name)
    samples: list[AraEvalSample] = []

    if hf_path:
        try:
            from datasets import load_dataset

            ds = None
            for split_name in ["test", "train", "validation"]:
                try:
                    ds = load_dataset(hf_path, split=split_name)
                    break
                except Exception:
                    continue

            if ds is None:
                ds = load_dataset(hf_path)
                if hasattr(ds, "keys"):
                    first_key = list(ds.keys())[0]
                    ds = ds[first_key]

            records = list(ds)
            if limit:
                records = records[:limit]
            for idx, raw in enumerate(records):
                samples.append(normalize_sample(raw, task_name, idx))
            return samples
        except Exception as err:
            logger.warning(f"Could not load HF dataset {hf_path}: {err}. Using mock fallback.")
            pass

    # Mock samples for fallback / offline testing
    mock_records = _get_mock_samples(task_name)
    if limit:
        mock_records = mock_records[:limit]
    for idx, raw in enumerate(mock_records):
        samples.append(normalize_sample(raw, task_name, idx))
    return samples


def _get_mock_samples(task_name: str) -> list[dict[str, Any]]:
    """Synthetic samples mimicking official AraEval datasets."""
    if task_name == "ara_ifeval":
        return [
            {
                "id": "ifeval_1",
                "question": "اكتب فقرة عن الذكاء الاصطناعي مع تضمين كلمة 'الخوارزميات' وتجنب استخدام كلمة 'الحاسوب'.",
                "instructions": [
                    {"type": "include_keyword", "keyword": "الخوارزميات"},
                    {"type": "exclude_keyword", "keyword": "الحاسوب"},
                ],
            }
        ]
    elif task_name == "ara_math":
        return [
            {
                "id": "math_1",
                "question": "إذا كان طول مستطيل 8 سم وعرضه 5 سم، فما هي مساحته؟",
                "options": {"A": "30", "B": "40", "C": "13", "D": "26"},
                "answer": "B",
            }
        ]
    elif task_name == "ara_truthfulqa":
        return [
            {
                "id": "truth_1",
                "question": "ما هي عاصمة المملكة العربية السعودية؟",
                "options": {"A": "جدة", "B": "الرياض", "C": "مكة المكرمة", "D": "الدمام"},
                "answer": "B",
            }
        ]
    return [
        {
            "id": f"{task_name}_sample_1",
            "question": "سؤال عام في العلوم واللغة العربية؟",
            "options": {"A": "خيار 1", "B": "خيار 2", "C": "خيار 3", "D": "خيار 4"},
            "answer": "A",
        }
    ]
