"""
Data loading for the SOTA pipeline.

Handles the real Arabic Reasoning dataset schema:
    {"id": "...", "source": "rlvr", "domain": "gsm8k|math|math_comp|logic|mmlu",
     "prompt": "...", "response": "", "answer": "",
     "metadata": {"grade_level": "...", "num_steps": N, "ground_truth_answer": <any>},
     "quality": {"score": ..., "arabic_purity": ...}}

Produces HuggingFace Datasets in conversational format for TRL's GRPOTrainer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset


def _serialize_ground_truth(gt: Any) -> str:
    if gt is None:
        return ""
    if isinstance(gt, bool):
        return "true" if gt else "false"
    if isinstance(gt, (int, float, np.integer, np.floating)):
        return str(gt.item() if isinstance(gt, (np.integer, np.floating)) else gt)
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False)
    return str(gt)

DOMAIN_MAP: dict[str, str] = {
    "gsm8k": "math",
    "math": "math",
    "math_comp": "math",
    "logic": "logic",
    "mmlu": "math",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _derive_difficulty(metadata: dict[str, Any]) -> str:
    num_steps = metadata.get("num_steps")
    if isinstance(num_steps, (int, float)) and num_steps:
        if num_steps <= 2:
            return "trivial"
        if num_steps <= 4:
            return "easy"
        if num_steps <= 6:
            return "medium"
        return "hard"
    grade = str(metadata.get("grade_level", "")).lower()
    if "grade_1" in grade or "grade_2" in grade or "grade_3" in grade:
        return "easy"
    if "grade_4" in grade or "grade_5" in grade:
        return "medium"
    if grade:
        return "hard"
    return "medium"


def _transform_coldstart_response(response: str) -> str:
    if "<answer>" in response and "</answer>" in response:
        return response
    match = re.search(r"####\s*(.+?)\s*$", response, re.MULTILINE)
    if match:
        answer_text = match.group(1).strip()
        response = re.sub(r"####\s*.+?\s*$", "", response, flags=re.MULTILINE).rstrip()
        return f"{response}\n<answer>{answer_text}</answer>"
    return response


def load_rlvr_dataset(
    path: str | Path,
    domain: str | None = None,
    system_prompt: str | None = None,
    validate: bool = False,
) -> Dataset:
    """Load a real Arabic Reasoning JSONL into a HF Dataset for TRL GRPOTrainer.

    Extracts ground_truth_answer from metadata, maps domains to reward-compatible
    types, derives difficulty from num_steps, and formats prompts conversationally.
    Samples with empty ground_truth_answer are filtered out.
    """
    records = _read_jsonl(path)

    prompts = []
    ground_truths = []
    domains = []
    puzzle_types = []
    difficulties = []
    sample_ids = []

    for raw in records:
        metadata = raw.get("metadata", {})
        gt = metadata.get("ground_truth_answer")
        raw_domain = raw.get("domain", "math")

        if gt is None or gt == "":
            continue

        reward_domain = DOMAIN_MAP.get(raw_domain, "math")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw["prompt"]})

        prompts.append(messages)
        ground_truths.append(_serialize_ground_truth(gt))
        domains.append(reward_domain)
        puzzle_types.append(raw_domain if reward_domain == "logic" else None)
        difficulties.append(_derive_difficulty(metadata))
        sample_ids.append(raw.get("id", ""))

    return Dataset.from_dict({
        "prompt": prompts,
        "ground_truth_answer": ground_truths,
        "domain": domains,
        "puzzle_type": puzzle_types,
        "difficulty_tag": difficulties,
        "sample_id": sample_ids,
    })


def load_cold_start_sft_dataset(
    path: str | Path,
    system_prompt: str | None = None,
) -> Dataset:
    """Load cold-start CoT data for SFT warm-up before GRPO.

    Transforms the `#### answer` format to `<answer>answer</answer>` so the
    model learns the same tag structure used during RLVR.

    Returns a dataset with a single 'messages' column containing full
    conversations [system, user, assistant] for direct SFTTrainer use.
    This avoids tokenization alignment warnings by letting SFTTrainer
    handle the chat template end-to-end.
    """
    records = _read_jsonl(path)
    all_messages = []
    for raw in records:
        response = raw.get("response", "")
        if not response:
            continue
        response = _transform_coldstart_response(response)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw["prompt"]})
        messages.append({"role": "assistant", "content": response})
        all_messages.append(messages)

    return Dataset.from_dict({"messages": all_messages})


def load_rlvr_with_curriculum(
    path: str | Path,
    system_prompt: str | None = None,
) -> Dataset:
    """Load RLVR data with difficulty_tag column for curriculum learning.

    Returns a Dataset with a 'difficulty_tag' column containing
    'trivial'/'easy'/'medium'/'hard' derived from metadata.num_steps.
    """
    return load_rlvr_dataset(path, system_prompt=system_prompt)
