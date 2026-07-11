"""
Module 1 — Dataset Loader & Schema Validator.

Load and validate the three dataset types:
  - Cold-Start CoT (Dataset 1)
  - RLVR Math (Dataset 2)
  - RLVR Logic (Dataset 3)
"""

import json
import re
from typing import TypedDict, NotRequired, Literal, Any


class SchemaValidationError(Exception):
    pass


class ColdStartSample(TypedDict):
    id: str
    domain: str
    difficulty: str
    prompt: str
    response: str


class RLVRSample(TypedDict):
    id: str
    domain: str
    difficulty_tag: str
    prompt: str
    ground_truth_answer: Any
    puzzle_type: NotRequired[str]


_VALID_DOMAINS = ("math", "logic")
_SHINGLE_SIZE = 3
_TAG_OPEN_THINK = "<think>"
_TAG_CLOSE_THINK = "</think>"
_TAG_OPEN_ANSWER = "<answer>"
_TAG_CLOSE_ANSWER = "</answer>"


def _sample_label(sample: dict) -> str:
    sid = sample.get("id", "")
    if isinstance(sid, str) and sid.strip():
        return repr(sid)
    return "<missing id>"


def validate_cold_start_sample(sample: dict) -> ColdStartSample:
    if not isinstance(sample, dict):
        raise SchemaValidationError(
            f"ColdStart sample is not a JSON object: {type(sample).__name__}"
        )
    label = _sample_label(sample)
    required = ("id", "domain", "difficulty", "prompt", "response")
    for field in required:
        if field not in sample:
            raise SchemaValidationError(
                f"ColdStart sample {label}: missing required field {field!r}"
            )
    for field in ("id", "domain", "difficulty", "prompt"):
        value = sample[field]
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(
                f"ColdStart sample {label}: field {field!r} must be a non-empty string"
            )
    response = sample["response"]
    if not isinstance(response, str):
        raise SchemaValidationError(
            f"ColdStart sample {label}: field 'response' must be a string"
        )
    for tag in (_TAG_OPEN_THINK, _TAG_CLOSE_THINK, _TAG_OPEN_ANSWER, _TAG_CLOSE_ANSWER):
        if tag not in response:
            raise SchemaValidationError(
                f"ColdStart sample {label}: response is missing required tag {tag!r}"
            )
    return {
        "id": sample["id"],
        "domain": sample["domain"],
        "difficulty": sample["difficulty"],
        "prompt": sample["prompt"],
        "response": sample["response"],
    }


def validate_rlvr_sample(sample: dict, domain: Literal["math", "logic"]) -> RLVRSample:
    if not isinstance(sample, dict):
        raise SchemaValidationError(
            f"RLVR sample is not a JSON object: {type(sample).__name__}"
        )
    if domain not in _VALID_DOMAINS:
        raise SchemaValidationError(
            f"domain must be 'math' or 'logic', got {domain!r}"
        )
    label = _sample_label(sample)
    required = ("id", "domain", "difficulty_tag", "prompt", "ground_truth_answer")
    for field in required:
        if field not in sample:
            raise SchemaValidationError(
                f"RLVR sample {label}: missing required field {field!r}"
            )
    for field in ("id", "domain", "difficulty_tag", "prompt"):
        value = sample[field]
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(
                f"RLVR sample {label}: field {field!r} must be a non-empty string"
            )
    sample_domain = sample["domain"]
    if sample_domain not in _VALID_DOMAINS:
        raise SchemaValidationError(
            f"RLVR sample {label}: domain field must be 'math' or 'logic', "
            f"got {sample_domain!r}"
        )
    ground_truth = sample["ground_truth_answer"]
    if ground_truth is None:
        raise SchemaValidationError(
            f"RLVR sample {label}: field 'ground_truth_answer' must not be null"
        )
    if domain == "logic":
        if "puzzle_type" not in sample:
            raise SchemaValidationError(
                f"RLVR sample {label}: logic samples require a 'puzzle_type' field"
            )
        puzzle_type = sample["puzzle_type"]
        if not isinstance(puzzle_type, str) or not puzzle_type.strip():
            raise SchemaValidationError(
                f"RLVR sample {label}: 'puzzle_type' must be a non-empty string"
            )
    result: RLVRSample = {
        "id": sample["id"],
        "domain": sample["domain"],
        "difficulty_tag": sample["difficulty_tag"],
        "prompt": sample["prompt"],
        "ground_truth_answer": sample["ground_truth_answer"],
    }
    if "puzzle_type" in sample:
        result["puzzle_type"] = sample["puzzle_type"]
    return result


def _read_samples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    stripped = content.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [item for item in parsed]
    samples: list[dict] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"Invalid JSON at line {lineno} in {path}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise SchemaValidationError(
                f"Entry at line {lineno} in {path} is not a JSON object"
            )
        samples.append(obj)
    return samples


def load_cold_start_dataset(path: str) -> list[ColdStartSample]:
    raw_samples = _read_samples(path)
    validated: list[ColdStartSample] = []
    seen_ids: set[str] = set()
    for raw in raw_samples:
        sample = validate_cold_start_sample(raw)
        if sample["id"] in seen_ids:
            raise SchemaValidationError(
                f"Duplicate id {sample['id']!r} within cold-start dataset"
            )
        seen_ids.add(sample["id"])
        validated.append(sample)
    return validated


def load_rlvr_dataset(path: str, domain: Literal["math", "logic"]) -> list[RLVRSample]:
    if domain not in _VALID_DOMAINS:
        raise SchemaValidationError(
            f"domain must be 'math' or 'logic', got {domain!r}"
        )
    raw_samples = _read_samples(path)
    validated: list[RLVRSample] = []
    seen_ids: set[str] = set()
    for raw in raw_samples:
        sample = validate_rlvr_sample(raw, domain)
        if sample["id"] in seen_ids:
            raise SchemaValidationError(
                f"Duplicate id {sample['id']!r} within RLVR dataset"
            )
        seen_ids.add(sample["id"])
        validated.append(sample)
    return validated


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _shingles(text: str, n: int = _SHINGLE_SIZE) -> set[tuple[str, ...]]:
    tokens = _tokenize(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def find_near_duplicate_prompts(
    coldstart_samples: list[ColdStartSample],
    rlvr_samples: list[RLVRSample],
    threshold: float = 0.8,
) -> list[tuple[str, str]]:
    coldstart_sets = [
        (str(sample.get("id", "")), _shingles(str(sample.get("prompt", ""))))
        for sample in coldstart_samples
    ]
    rlvr_sets = [
        (str(sample.get("id", "")), _shingles(str(sample.get("prompt", ""))))
        for sample in rlvr_samples
    ]
    duplicates: list[tuple[str, str]] = []
    for cs_id, cs_shingles in coldstart_sets:
        for rl_id, rl_shingles in rlvr_sets:
            if _jaccard(cs_shingles, rl_shingles) > threshold:
                duplicates.append((cs_id, rl_id))
    return duplicates
