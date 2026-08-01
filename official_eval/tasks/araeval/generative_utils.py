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
    "araeval_ifeval",
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
        "gpu_memory_utilization": 0.88,
        "max_num_seqs": 64,
    },
    # AraMath: long chain-of-thought math reasoning
    "araeval_aramath": {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "gpu_memory_utilization": 0.88,
        "max_num_seqs": 32,
    },
    # AraPro: long medical/science passages + reasoning
    "araeval_arapro": {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "gpu_memory_utilization": 0.80,
        "max_num_seqs": 16,
    },
    # IFEval: full instruction-following generation
    "araeval_ifeval": {
        "max_new_tokens": 1280,
        "temperature": 0.0,
        "gpu_memory_utilization": 0.88,
        "max_num_seqs": 32,
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
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)

_ARABIC_TO_LATIN = {
    "أ": "A",
    "ب": "B",
    "ج": "C",
    "د": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
}


def strip_thinking_tags(text: str) -> str:
    """Remove all <think>...</think> blocks (including unclosed <think>...) from generated text."""
    # First remove closed <think>...</think> blocks
    stripped = _THINK_CLOSED_RE.sub("", text)
    # If <think> is still present (unclosed), strip from <think> to end of text
    if "<think>" in stripped:
        stripped = _THINK_OPEN_RE.sub("", stripped)
    return stripped


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
# All choice tokens (Latin A-D and Arabic أ/ب/ج/د)
_CHOICE_PATTERN = r"([A-Da-dأبجد])"

# Layer 1: Explicit answer statement patterns (e.g. الإجابة: A, الإجابة هي (أ), Answer: B)
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:الإجابة|الاجابة|الجواب|الإجابة\s+الصحيحة|answer)"
    r"[\s:：]*(?:هي|هو|الصحيحة?|is|are)?\s*[:：]?\s*"
    r"[\*\(\[\"\']*" + _CHOICE_PATTERN + r"[\*\)\]\"\']*",
    re.IGNORECASE,
)

# Layer 2: "option/choice is X" patterns (e.g. الخيار الصحيح هو أ, option (B))
_OPTION_RE = re.compile(
    r"(?:الخيار|الاختيار|option|choice)\s+(?:الصحيح|الصحيحة?)?\s*(?:هو|هي)?\s*"
    r"[\*\(\[\"\']*" + _CHOICE_PATTERN + r"[\*\)\]\"\']*",
    re.IGNORECASE,
)

# Layer 3: Markdown / bracketed letter patterns (e.g. **A**, [B], (أ), **C.**)
_BRACKETED_RE = re.compile(
    r"(?:\*\*|__|\[|\()[\s]*" + _CHOICE_PATTERN + r"[\s]*[\.\)]*[\s]*(?:\*\*|__|\]|\))"
)

# Layer 4: Standalone letter at end of text
_END_LETTER_RE = re.compile(_CHOICE_PATTERN + r"\s*[.。)]*\s*$")

# Layer 5: Any choice letter (last occurrence)
_ANY_LETTER_RE = re.compile(r"(?<!\w)" + _CHOICE_PATTERN + r"(?!\w)")


def _extract_from_text(search_text: str) -> str | None:
    if not search_text or not search_text.strip():
        return None

    # Layer 1: All explicit answer statements — take the LAST one
    matches = list(_EXPLICIT_ANSWER_RE.finditer(search_text))
    if matches:
        raw = matches[-1].group(1)
        return _ARABIC_TO_LATIN.get(raw)

    # Layer 2: All option/choice references — take the LAST one
    matches = list(_OPTION_RE.finditer(search_text))
    if matches:
        raw = matches[-1].group(1)
        return _ARABIC_TO_LATIN.get(raw)

    # Layer 3: Markdown / bracketed letter patterns — take the LAST one
    matches = list(_BRACKETED_RE.finditer(search_text))
    if matches:
        raw = matches[-1].group(1)
        return _ARABIC_TO_LATIN.get(raw)

    # Layer 4: Letter at end of text
    match = _END_LETTER_RE.search(search_text)
    if match:
        raw = match.group(1)
        return _ARABIC_TO_LATIN.get(raw)

    # Layer 5: Last standalone choice letter
    all_letters = _ANY_LETTER_RE.findall(search_text)
    if all_letters:
        raw = all_letters[-1]
        return _ARABIC_TO_LATIN.get(raw)

    return None


def extract_answer(text: str) -> str | None:
    """Extract the answer letter (A/B/C/D) from model-generated text.

    First tries extracting outside <think> tags. If no answer is found (e.g.
    the model put the answer inside <think> or <think> was unclosed), falls
    back to extracting from the raw text.
    """
    if not text or not text.strip():
        return None

    # First pass: search outside <think> tags
    clean = strip_thinking_tags(text).strip()
    ans = _extract_from_text(clean)
    if ans is not None:
        return ans

    # Fallback pass: search raw text (handles unclosed <think> or answers inside <think>)
    return _extract_from_text(text)


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
