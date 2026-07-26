"""
Dataset loader for English baseline benchmarks (IFEval, MMLU, HellaSwag, GSM8K).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from english_eval.config import ENGLISH_DATASETS, MMLU_CORE_SUBJECTS

logger = logging.getLogger(__name__)


@dataclass
class EnglishEvalSample:
    sample_id: str
    task_name: str
    prompt: str
    gold_answer: str
    options: dict[str, str] = field(default_factory=dict)
    instruction_ids: list[str] = field(default_factory=list)
    kwargs_list: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_ifeval_dataset(limit: Optional[int] = None) -> list[EnglishEvalSample]:
    """Load Google IFEval dataset."""
    from datasets import load_dataset

    print("Loading English IFEval (google/ifeval)...", flush=True)
    try:
        ds = load_dataset("google/ifeval", split="train")
    except Exception:
        try:
            ds = load_dataset("wisecube/ifeval", split="train")
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face IFEval: {e}")
            return []

    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        key = str(row.get("key", i))
        prompt = row.get("prompt", "")
        inst_ids = row.get("instruction_id_list", [])
        kwargs = row.get("kwargs", [])

        samples.append(
            EnglishEvalSample(
                sample_id=f"ifeval_{key}",
                task_name="ifeval",
                prompt=prompt,
                gold_answer="strict_and_loose_rules",
                instruction_ids=inst_ids,
                kwargs_list=kwargs,
                metadata={"key": key},
            )
        )
    return samples


def load_mmlu_dataset(limit: Optional[int] = None, subjects: Optional[list[str]] = None) -> list[EnglishEvalSample]:
    """Load English MMLU dataset across subjects."""
    from datasets import load_dataset

    target_subjects = subjects or MMLU_CORE_SUBJECTS
    print(f"Loading English MMLU ({len(target_subjects)} subjects)...", flush=True)
    samples = []

    per_subj_limit = None
    if limit:
        per_subj_limit = max(1, limit // len(target_subjects))

    option_keys = ["A", "B", "C", "D"]

    for subj in target_subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            try:
                ds = load_dataset("cais/mmlu", subj, split="val")
            except Exception as e:
                logger.warning(f"Could not load MMLU subject {subj}: {e}")
                continue

        subj_count = 0
        for row in ds:
            if per_subj_limit and subj_count >= per_subj_limit:
                break

            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_idx = row.get("answer", 0)

            options = {k: str(v) for k, v in zip(option_keys, choices)}
            gold = option_keys[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(option_keys) else str(answer_idx)

            options_formatted = "\n".join([f"{k}. {v}" for k, v in options.items()])
            prompt = (
                f"The following is a multiple-choice question about {subj.replace('_', ' ')}.\n\n"
                f"Question: {question}\n\n"
                f"{options_formatted}\n\n"
                "Please select the correct option letter (A, B, C, or D).\nAnswer:"
            )

            samples.append(
                EnglishEvalSample(
                    sample_id=f"mmlu_{subj}_{subj_count}",
                    task_name="mmlu",
                    prompt=prompt,
                    gold_answer=gold,
                    options=options,
                    metadata={"subject": subj},
                )
            )
            subj_count += 1
            if limit and len(samples) >= limit:
                return samples

    return samples


def load_hellaswag_dataset(limit: Optional[int] = None) -> list[EnglishEvalSample]:
    """Load English HellaSwag dataset."""
    from datasets import load_dataset

    print("Loading English HellaSwag (Rowan/hellaswag)...", flush=True)
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
    except Exception:
        try:
            ds = load_dataset("Rowan/hellaswag", split="train")
        except Exception as e:
            logger.warning(f"Failed to load HellaSwag: {e}")
            return []

    option_keys = ["A", "B", "C", "D"]
    samples = []

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        ctx = row.get("ctx", "") or (row.get("ctx_a", "") + " " + row.get("ctx_b", ""))
        endings = row.get("endings", [])
        label_raw = row.get("label", 0)

        try:
            label_idx = int(label_raw)
        except ValueError:
            label_idx = 0

        gold = option_keys[label_idx] if label_idx < len(option_keys) else "A"
        options = {k: str(v) for k, v in zip(option_keys, endings)}
        options_formatted = "\n".join([f"{k}. {v}" for k, v in options.items()])

        prompt = (
            "Choose the most logical continuation for the situation described below:\n\n"
            f"Context: {ctx}\n\n"
            f"Options:\n{options_formatted}\n\n"
            "Please select the correct option letter (A, B, C, or D).\nAnswer:"
        )

        samples.append(
            EnglishEvalSample(
                sample_id=f"hellaswag_{i}",
                task_name="hellaswag",
                prompt=prompt,
                gold_answer=gold,
                options=options,
                metadata={"activity_label": row.get("activity_label", "")},
            )
        )

    return samples


def load_gsm8k_dataset(limit: Optional[int] = None) -> list[EnglishEvalSample]:
    """Load English GSM8K dataset."""
    from datasets import load_dataset

    print("Loading English GSM8K (gsm8k)...", flush=True)
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}")
        return []

    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        question = row.get("question", "")
        answer = row.get("answer", "")

        prompt = (
            "Solve the following mathematical word problem step by step, and provide your final numerical answer after '#### '.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        samples.append(
            EnglishEvalSample(
                sample_id=f"gsm8k_{i}",
                task_name="gsm8k",
                prompt=prompt,
                gold_answer=answer,
                metadata={},
            )
        )

    return samples


def load_english_task(task_name: str, limit: Optional[int] = None, mmlu_subjects: Optional[list[str]] = None) -> list[EnglishEvalSample]:
    """Load specified English benchmark dataset."""
    t_clean = task_name.lower().strip()
    if t_clean == "ifeval":
        return load_ifeval_dataset(limit=limit)
    elif t_clean == "mmlu":
        return load_mmlu_dataset(limit=limit, subjects=mmlu_subjects)
    elif t_clean == "hellaswag":
        return load_hellaswag_dataset(limit=limit)
    elif t_clean == "gsm8k":
        return load_gsm8k_dataset(limit=limit)
    else:
        logger.warning(f"Unknown English evaluation task: {task_name}")
        return []
