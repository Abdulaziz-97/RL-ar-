"""Generative evaluation utilities for AraEval benchmarks.

This module provides answer extraction, prompt building, grading, and
checkpoint management for generation-based (not log-likelihood) evaluation
of Arabic language models on the official AraEval task suite.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Task list: all MCQ tasks (IFEval is already generation-based in official eval)
# ---------------------------------------------------------------------------
GENERATIVE_TASKS = [
    "araeval_ien_mcq",
    "araeval_ien_tf",
    "araeval_aramath",
    "araeval_etec",
    "araeval_arapro",
    "araeval_truthfulqa",
]

_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ---------------------------------------------------------------------------
# Per-task generation hyperparameter profiles
# ---------------------------------------------------------------------------
GENERATION_PROFILES: dict[str, dict[str, Any]] = {
    # Short MCQ tasks: long reasoning chain + letter answer
    "gen_mcq": {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "batch_size": 16,
        "gpu_memory_utilization": 0.75,
    },
    # AraMath: long chain-of-thought math reasoning
    "araeval_aramath": {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "batch_size": 8,
        "gpu_memory_utilization": 0.75,
    },
    # AraPro: long medical/science passages + reasoning
    "araeval_arapro": {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "batch_size": 4,
        "gpu_memory_utilization": 0.50,
    },
    # IFEval: full instruction-following generation
    "araeval_ifeval": {
        "max_new_tokens": 1280,
        "temperature": 0.0,
        "batch_size": 8,
        "gpu_memory_utilization": 0.85,
    },
}

_TASK_TO_PROFILE: dict[str, str] = {
    "araeval_ien_mcq": "gen_mcq",
    "araeval_ien_tf": "gen_mcq",
    "araeval_etec": "gen_mcq",
    "araeval_truthfulqa": "gen_mcq",
    "araeval_aramath": "araeval_aramath",
    "araeval_arapro": "araeval_arapro",
    "araeval_ifeval": "araeval_ifeval",
}


def get_generation_profile(task: str) -> dict[str, Any]:
    """Return the generation hyperparameter profile for a given task."""
    profile_key = _TASK_TO_PROFILE.get(task, "gen_mcq")
    return GENERATION_PROFILES.get(profile_key, GENERATION_PROFILES["gen_mcq"])


# ---------------------------------------------------------------------------
# Thinking tag handling
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove all <think>...</think> blocks from generated text."""
    return _THINK_RE.sub("", text)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
# Layer 1: Explicit answer statement patterns
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:الإجابة|الاجابة|الجواب|الإجابة\s+الصحيحة|answer)"
    r"[\s:：]*(?:هي|هو|الصحيحة?|is|are)?\s*[:：]?\s*"
    r"([A-Da-d])",
    re.IGNORECASE,
)

# Layer 2: "option/choice is X" patterns
_OPTION_RE = re.compile(
    r"(?:الخيار|الاختيار|option|choice)\s+(?:الصحيح|الصحيحة?)?\s*(?:هو|هي)?\s*"
    r"([A-Da-d])",
    re.IGNORECASE,
)

# Layer 3: Standalone letter at end of text
_END_LETTER_RE = re.compile(r"([A-Da-d])\s*[.。)]*\s*$")

# Layer 4: Any A-D letter (last occurrence)
_ANY_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def extract_answer(text: str) -> str | None:
    """Extract the answer letter (A/B/C/D) from model-generated text.

    Uses a multi-layer fallback strategy:
    1. Explicit answer statement (الإجابة: X) — takes LAST match
    2. Option/choice reference (الخيار الصحيح هو X) — takes LAST match
    3. Letter at the end of text
    4. Last standalone A-D letter in text

    Returns the uppercase letter or None if extraction fails.
    """
    if not text or not text.strip():
        return None

    # Strip thinking tags first
    clean = strip_thinking_tags(text).strip()
    if not clean:
        return None

    # Layer 1: All explicit answer statements — take the LAST one (final answer)
    matches = list(_EXPLICIT_ANSWER_RE.finditer(clean))
    if matches:
        return matches[-1].group(1).upper()

    # Layer 2: All option/choice references — take the LAST one
    matches = list(_OPTION_RE.finditer(clean))
    if matches:
        return matches[-1].group(1).upper()

    # Layer 3: Letter at end of text
    match = _END_LETTER_RE.search(clean)
    if match:
        return match.group(1).upper()

    # Layer 4: Last standalone A-D letter
    all_letters = _ANY_LETTER_RE.findall(clean)
    if all_letters:
        return all_letters[-1].upper()

    return None


# ---------------------------------------------------------------------------
# Answer grading
# ---------------------------------------------------------------------------
def grade_answer(extracted: str | None, gold_index: int) -> bool:
    """Grade an extracted answer against the gold 0-based index.

    Args:
        extracted: The extracted letter (A/B/C/D) or None if extraction failed.
        gold_index: The 0-based index of the correct answer (0=A, 1=B, 2=C, 3=D).

    Returns:
        True if correct, False otherwise.
    """
    if extracted is None:
        return False
    expected = _LABELS[gold_index]
    return extracted.upper() == expected


# ---------------------------------------------------------------------------
# Generative MCQ prompt building
# ---------------------------------------------------------------------------
def build_generative_prompt(
    question: str,
    choices: list[str],
    *,
    context: str = "",
    instruction: str = "",
) -> str:
    """Build a generation prompt for an MCQ question.

    The prompt is formatted to match the official AraEval MCQ format
    but designed for generation mode (the model generates its answer).

    Args:
        question: The question text.
        choices: List of choice texts.
        context: Optional context/passage.
        instruction: Optional instruction prefix (used by TruthfulQA).

    Returns:
        Formatted prompt string ending with 'الإجابة:'.
    """
    lines: list[str] = []
    if instruction.strip():
        lines.append(instruction.strip())
    if context.strip():
        lines.append(f"السياق: {context.strip()}")
    lines.append(f"السؤال: {question.strip()}")
    labels = _LABELS[: len(choices)]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice.strip()}")
    lines.append("الإجابة:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def new_generative_checkpoint(
    model: str,
    adapter_path: str | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Create a new generative evaluation checkpoint."""
    return {
        "version": 1,
        "evaluation_mode": "generative",
        "model": model,
        "adapter_path": adapter_path,
        "enable_thinking": enable_thinking,
        "tasks": list(GENERATIVE_TASKS),
        "completed": {},
    }


def save_generative_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Atomically save a checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_generative_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint from disk. Returns None if file doesn't exist."""
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pending_generative_tasks(checkpoint: dict[str, Any]) -> list[str]:
    """Return the list of tasks that still need to be evaluated."""
    completed = checkpoint.get("completed", {})
    return [task for task in GENERATIVE_TASKS if task not in completed]


def summarize_generative_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Generate a summary report from a completed checkpoint."""
    completed = checkpoint.get("completed", {})

    task_results: dict[str, Any] = {}
    total_correct = 0
    total_questions = 0
    total_extraction_failures = 0
    total_seconds = 0.0

    for task, data in completed.items():
        task_results[task] = {
            "correct": data["correct"],
            "total": data["total"],
            "accuracy": data["accuracy"],
            "extraction_failures": data.get("extraction_failures", 0),
            "seconds": data.get("seconds", 0.0),
        }
        total_correct += data["correct"]
        total_questions += data["total"]
        total_extraction_failures += data.get("extraction_failures", 0)
        total_seconds += data.get("seconds", 0.0)

    overall_accuracy = (
        100.0 * total_correct / total_questions if total_questions > 0 else 0.0
    )

    return {
        "evaluation_mode": "generative",
        "model": checkpoint.get("model"),
        "adapter_path": checkpoint.get("adapter_path"),
        "enable_thinking": checkpoint.get("enable_thinking"),
        "task_results": task_results,
        "overall_generative_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "total_extraction_failures": total_extraction_failures,
        "total_seconds": total_seconds,
    }
