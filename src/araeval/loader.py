"""
Dataset loading and schema normalization for AraEval benchmarks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from araeval.config import ARAEVAL_DATASETS

logger = logging.getLogger(__name__)


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
    sample_id = str(raw.get("id") or raw.get("IF_id") or raw.get("question_id") or raw.get("key") or f"{task_name}_{idx}")

    instructions = []
    # Handle AraIFEval schema
    if "instruction_following_prompt" in raw and isinstance(raw["instruction_following_prompt"], dict):
        ifp = raw["instruction_following_prompt"]
        question = str(ifp.get("prompt") or ifp.get("question") or "").strip()
        cats = ifp.get("categories") or []
        instructions = [{"type": cat} for cat in cats] if isinstance(cats, list) else []
    else:
        question = str(
            raw.get("question")
            or raw.get("Question")
            or raw.get("prompt")
            or raw.get("input")
            or ""
        ).strip()
        instructions = raw.get("instruction_list") or raw.get("instructions") or []

    # Handle LC-Eval context
    context = str(raw.get("context") or "").strip()
    if context:
        question = f"السياق: {context}\n\nالسؤال: {question}"

    # Handle options and gold answer
    options: dict[str, str] = {}

    # Handle AraTruthfulQA mc1_targets schema if present
    if "mc1_targets" in raw and isinstance(raw["mc1_targets"], dict):
        mc1 = raw["mc1_targets"]
        choices = mc1.get("choices") or []
        labels = mc1.get("labels") or []
        labels_alpha = ["A", "B", "C", "D", "E", "F", "G", "H"]
        options = {labels_alpha[i]: str(c) for i, c in enumerate(choices) if i < len(labels_alpha)}
        gold = "A"
        if isinstance(labels, list):
            for i, l in enumerate(labels):
                if l == 1 and i < len(labels_alpha):
                    gold = labels_alpha[i]
                    break
    else:
        raw_options = (
            raw.get("options")
            or raw.get("Options")
            or raw.get("choices")
            or raw.get("Choices")
        )
        if isinstance(raw_options, str):
            try:
                raw_options = json.loads(raw_options)
            except Exception:
                try:
                    import ast
                    raw_options = ast.literal_eval(raw_options)
                except Exception:
                    pass

        if isinstance(raw_options, dict):
            options = {str(k).upper(): str(v) for k, v in raw_options.items()}
        elif isinstance(raw_options, list):
            labels_alpha = ["A", "B", "C", "D", "E", "F"]
            options = {labels_alpha[i]: str(opt) for i, opt in enumerate(raw_options) if i < len(labels_alpha)}
        elif any(k in raw for k in ["option_a", "A", "option_A", "choice1", "choice_1"]):
            for i, lbl in enumerate(["A", "B", "C", "D"]):
                val = (
                    raw.get(lbl)
                    or raw.get(f"option_{lbl.lower()}")
                    or raw.get(f"option_{lbl}")
                    or raw.get(f"choice{i + 1}")
                    or raw.get(f"choice_{i + 1}")
                )
                if val:
                    options[lbl] = str(val)

        # Canonical Gold Answer Resolution (fair & exact)
        raw_label = raw.get("label") if raw.get("label") is not None else raw.get("Label")
        raw_answer = (
            raw.get("answer")
            if raw.get("answer") is not None
            else (
                raw.get("Answer")
                if raw.get("Answer") is not None
                else (
                    raw.get("target")
                    if raw.get("target") is not None
                    else (
                        raw.get("Target")
                        if raw.get("Target") is not None
                        else (raw.get("gold") if raw.get("gold") is not None else raw.get("Gold"))
                    )
                )
            )
        )

        gold = ""
        if raw_label is not None:
            lbl_str = str(raw_label).strip().upper()
            if lbl_str in options:
                gold = lbl_str
            elif lbl_str.isdigit() and options:
                idx_int = int(lbl_str)
                keys = list(options.keys())
                if 1 <= idx_int <= len(keys):
                    gold = keys[idx_int - 1]
                elif 0 <= idx_int < len(keys):
                    gold = keys[idx_int]

        if not gold and raw_answer is not None:
            ans_str = str(raw_answer).strip().upper()
            if ans_str in options:
                gold = ans_str
            elif ans_str.isdigit() and options:
                idx_int = int(ans_str)
                keys = list(options.keys())
                if 1 <= idx_int <= len(keys):
                    gold = keys[idx_int - 1]
                elif 0 <= idx_int < len(keys):
                    gold = keys[idx_int]
            else:
                for opt_key, opt_text in options.items():
                    if opt_text and opt_text.strip().lower() == str(raw_answer).strip().lower():
                        gold = opt_key
                        break

    if options and not question.startswith("السؤال:"):
        prompt = format_mcq_prompt(question, options)
    else:
        prompt = question or f"السؤال: {task_name} sample {idx}"

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
