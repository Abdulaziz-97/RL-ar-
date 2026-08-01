"""
Data loading for the pipeline.

Supports:
  1. Production records with ``answer_spec``, ``family_id``, ``partition``.
  2. Legacy v2 Arabic Reasoning schema (clearly labeled adapter).

MMLU records (domain="mmlu") are excluded from RLVR training.
Symbolic answer_spec types are rejected fail-closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from rlvr_contracts.answer_spec import (
    AnswerSpecError,
    infer_answer_spec_from_legacy,
    parse_answer_spec,
    reject_symbolic,
)

DOMAIN_MAP: dict[str, str] = {
    "gsm8k": "math",
    "math": "math",
    "math_comp": "math",
    "logic": "logic",
    "ifeval_multiconstraint": "instruction",
}


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
    explicit = metadata.get("difficulty_tag")
    if explicit in {"trivial", "easy", "medium", "hard"}:
        return explicit
    emp = metadata.get("empirical_difficulty") or metadata.get("difficulty_band")
    if isinstance(emp, dict):
        emp = emp.get("band")
    if isinstance(emp, str) and emp:
        return emp
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


def _extract_answer_spec(raw: dict[str, Any]) -> dict[str, Any] | None:
    metadata = raw.get("metadata") or {}
    spec = raw.get("answer_spec") or metadata.get("answer_spec")
    if spec is None:
        return None
    if isinstance(spec, dict) and spec.get("type") == "logic_json":
        canon = spec.get("canonical")
        if isinstance(canon, (int, float)):
            spec["type"] = "integer"
            spec["canonical"] = int(round(canon))
        elif isinstance(canon, str):
            c_str = canon.strip()
            if c_str.isdigit():
                spec["type"] = "integer"
                spec["canonical"] = int(c_str)
            elif not (c_str.startswith("{") or c_str.startswith("[")):
                try:
                    spec["type"] = "integer"
                    spec["canonical"] = int(round(float(c_str)))
                except ValueError:
                    spec["canonical"] = json.dumps({"ans": c_str})
    parsed = parse_answer_spec(spec)
    reject_symbolic(parsed.type)
    return parsed.to_dict()


def _is_production_record(raw: dict[str, Any]) -> bool:
    return bool(raw.get("answer_spec") or (raw.get("metadata") or {}).get("answer_spec"))


def load_production_dataset(
    path: str | Path,
    system_prompt: str | None = None,
    *,
    require_answer_spec: bool = True,
    allow_legacy_fallback: bool = False,
    allowed_partitions: set[str] | None = None,
) -> Dataset:
    """Load production Problem / RLVR Prompt records with answer_spec."""
    records = _read_jsonl(path)
    prompts: list[Any] = []
    ground_truths: list[str] = []
    domains: list[str] = []
    puzzle_types: list[Any] = []
    difficulties: list[str] = []
    sample_ids: list[str] = []
    family_ids: list[str] = []
    partitions: list[str] = []
    answer_specs: list[str] = []
    verifier_versions: list[str] = []

    for raw in records:
        metadata = raw.get("metadata") or {}
        partition = str(raw.get("partition") or metadata.get("partition") or "")
        # Normalize legacy partition labels used by shipped V4 corpora.
        if partition in {"", "train", "rlvr"}:
            partition = "rlvr_train"
        elif partition == "sft":
            partition = "sft_train"
        if allowed_partitions is not None and partition not in allowed_partitions:
            continue
        raw_domain = raw.get("domain", "math")
        if raw_domain == "mmlu":
            continue

        try:
            spec = _extract_answer_spec(raw)
        except AnswerSpecError as exc:
            if not allow_legacy_fallback:
                raise AnswerSpecError(
                    f"rejecting record {raw.get('problem_id') or raw.get('id')}: {exc}"
                ) from exc
            continue

        gt = None
        if spec is not None:
            gt = spec.get("canonical")
        else:
            gt = metadata.get("ground_truth_answer", raw.get("ground_truth_answer"))
            if gt in (None, "") and require_answer_spec and not allow_legacy_fallback:
                continue
            if allow_legacy_fallback and gt not in (None, ""):
                try:
                    inferred = infer_answer_spec_from_legacy(
                        gt, domain=DOMAIN_MAP.get(raw_domain, "math")
                    )
                    spec = inferred.to_dict()
                    gt = inferred.canonical
                except AnswerSpecError:
                    continue
            elif gt in (None, ""):
                continue

        if spec is None and require_answer_spec:
            continue

        reward_domain = DOMAIN_MAP.get(raw_domain, "math")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        prompt_text = raw.get("prompt") or raw.get("problem_text") or ""
        messages.append({"role": "user", "content": prompt_text})

        prompts.append(messages)
        ground_truths.append(_serialize_ground_truth(gt))
        domains.append(reward_domain)
        puzzle_types.append(raw_domain if reward_domain == "logic" else None)
        difficulty_fields = {**raw, **metadata}
        difficulties.append(_derive_difficulty(difficulty_fields))
        sample_ids.append(str(raw.get("sample_id") or raw.get("problem_id") or raw.get("id") or raw.get("id_") or ""))
        family_ids.append(str(raw.get("family_id") or metadata.get("family_id") or ""))
        partitions.append(partition)
        answer_specs.append(json.dumps(spec, ensure_ascii=False) if spec else "")
        verifier_versions.append(
            str(
                raw.get("verifier_version")
                or metadata.get("verifier_version")
                or (spec or {}).get("verifier_artifact")
                or ""
            )
        )

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "ground_truth_answer": ground_truths,
            "domain": domains,
            "puzzle_type": puzzle_types,
            "difficulty_tag": difficulties,
            "sample_id": sample_ids,
            "family_id": family_ids,
            "partition": partitions,
            "answer_spec": answer_specs,
            "verifier_version": verifier_versions,
        }
    )


def load_rlvr_dataset_legacy_v2(
    path: str | Path,
    domain: str | None = None,
    system_prompt: str | None = None,
    validate: bool = False,
) -> Dataset:
    """LEGACY-V2 ADAPTER — migration only.

    Prefer :func:`load_production_dataset` for new corpora. This adapter keeps
    older GSM/logic JSONL trainable while production packs migrate to answer_spec.
    """
    records = _read_jsonl(path)

    prompts = []
    ground_truths = []
    domains = []
    puzzle_types = []
    difficulties = []
    sample_ids = []
    answer_specs = []

    for raw in records:
        metadata = raw.get("metadata", {})
        gt = metadata.get("ground_truth_answer")
        raw_domain = raw.get("domain", "math")
        if validate and raw_domain not in DOMAIN_MAP and raw_domain != "mmlu":
            raise ValueError(
                f"legacy record {raw.get('id', '')!r} has unsupported domain {raw_domain!r}"
            )
        reward_domain = DOMAIN_MAP.get(raw_domain, "math")

        if gt is None or gt == "":
            if validate:
                raise ValueError(
                    f"legacy record {raw.get('id', '')!r} has no ground_truth_answer"
                )
            continue
        if raw_domain == "mmlu":
            continue
        if domain is not None and reward_domain != domain:
            continue
        if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
            if validate:
                raise ValueError(f"legacy record {raw.get('id', '')!r} has no prompt")
            continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw["prompt"]})

        spec_json = ""
        try:
            if _is_production_record(raw):
                spec = _extract_answer_spec(raw)
                if spec:
                    spec_json = json.dumps(spec, ensure_ascii=False)
                    gt = spec["canonical"]
            else:
                inferred = infer_answer_spec_from_legacy(gt, domain=reward_domain)
                spec_json = json.dumps(inferred.to_dict(), ensure_ascii=False)
        except AnswerSpecError:
            # Keep legacy row without answer_spec; reward falls back to verify_legacy.
            spec_json = ""

        prompts.append(messages)
        ground_truths.append(_serialize_ground_truth(gt))
        domains.append(reward_domain)
        puzzle_types.append(raw_domain if reward_domain == "logic" else None)
        difficulties.append(_derive_difficulty(metadata))
        sample_ids.append(str(raw.get("sample_id") or raw.get("problem_id") or raw.get("id") or raw.get("id_") or ""))
        answer_specs.append(spec_json)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "ground_truth_answer": ground_truths,
            "domain": domains,
            "puzzle_type": puzzle_types,
            "difficulty_tag": difficulties,
            "sample_id": sample_ids,
            "answer_spec": answer_specs,
        }
    )


def load_rlvr_dataset(
    path: str | Path,
    domain: str | None = None,
    system_prompt: str | None = None,
    validate: bool = False,
    *,
    production: bool | None = None,
    allowed_partitions: set[str] | None = None,
    fail_closed: bool | None = None,
) -> Dataset:
    """Load RLVR data.

    Auto-detects production records (answer_spec present). Set
    ``production=True`` to require answer_spec, or ``False`` to force legacy-v2.

    When ``fail_closed`` is True (default when RLVR_V4_PRODUCTION=1), reject
    majority-legacy fallback and require ``rlvr_train`` partition rows with
    answer_spec / problem_id / family_id.
    """
    import os

    if fail_closed is None:
        fail_closed = os.environ.get("RLVR_V4_PRODUCTION", "").strip() in {"1", "true", "TRUE"}

    records = _read_jsonl(path)
    if not records:
        return Dataset.from_dict(
            {
                "prompt": [],
                "ground_truth_answer": [],
                "domain": [],
                "puzzle_type": [],
                "difficulty_tag": [],
                "sample_id": [],
                "answer_spec": [],
            }
        )

    if fail_closed:
        production = True
        if allowed_partitions is None:
            allowed_partitions = {"rlvr_train"}

    if production is True:
        ds = load_production_dataset(
            path,
            system_prompt=system_prompt,
            require_answer_spec=True,
            allow_legacy_fallback=not fail_closed,
            allowed_partitions=allowed_partitions,
        )
        if fail_closed and len(ds) == 0:
            raise ValueError(
                f"fail-closed V4 load produced 0 rows from {path}; "
                "expected rlvr_train rows with answer_spec"
            )
        if fail_closed:
            missing_ids = sum(1 for sid in ds["sample_id"] if not sid)
            missing_fam = (
                sum(1 for fid in ds["family_id"] if not fid)
                if "family_id" in ds.column_names
                else len(ds)
            )
            if missing_ids or missing_fam:
                raise ValueError(
                    f"fail-closed V4 data missing identifiers: "
                    f"empty sample_id={missing_ids}, empty family_id={missing_fam}"
                )
        return ds
    if production is False:
        return load_rlvr_dataset_legacy_v2(
            path, domain=domain, system_prompt=system_prompt, validate=validate
        )
    # Auto: if a majority of rows have answer_spec, treat as production.
    prod_hits = sum(1 for r in records if _is_production_record(r))
    if prod_hits > len(records) / 2:
        return load_production_dataset(
            path,
            system_prompt=system_prompt,
            require_answer_spec=True,
            allow_legacy_fallback=True,
            allowed_partitions=allowed_partitions,
        )
    return load_rlvr_dataset_legacy_v2(
        path, domain=domain, system_prompt=system_prompt, validate=validate
    )


def load_cold_start_sft_dataset(
    path: str | Path | list[str | Path],
    system_prompt: str | None = None,
    *,
    stratify_by_domain: bool = True,
    seed: int = 42,
) -> Dataset:
    """Load cold-start CoT data for SFT warm-up before GRPO.

    ``path`` may be a single JSONL or a list of corpora (core + sidecars).
    When ``stratify_by_domain`` is true, rows are shuffled with a round-robin
    mix across domains so small sidecars are not clumped at the end.
    """
    paths = path if isinstance(path, list) else [path]
    records: list[dict[str, Any]] = []
    for p in paths:
        records.extend(_read_jsonl(p))

    prepared: list[tuple[str, list[dict[str, str]]]] = []
    for raw in records:
        response = raw.get("response") or raw.get("trace") or ""
        if not response:
            continue
        try:
            if _is_production_record(raw):
                _extract_answer_spec(raw)
        except AnswerSpecError as exc:
            raise AnswerSpecError(
                f"SFT ingest rejected {raw.get('problem_id') or raw.get('id')}: {exc}"
            ) from exc
        response = _transform_coldstart_response(response)
        prompt = raw.get("prompt") or raw.get("problem_text") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        domain = str(raw.get("domain") or "math")
        prepared.append((domain, messages))

    if stratify_by_domain and prepared:
        rng = np.random.default_rng(seed)
        by_domain: dict[str, list[list[dict[str, str]]]] = {}
        for domain, messages in prepared:
            by_domain.setdefault(domain, []).append(messages)
        for domain in by_domain:
            rng.shuffle(by_domain[domain])
        # Round-robin across domains proportional to remaining counts.
        mixed: list[list[dict[str, str]]] = []
        domains = sorted(by_domain.keys())
        while any(by_domain[d] for d in domains):
            rng.shuffle(domains)
            for domain in domains:
                bucket = by_domain[domain]
                if bucket:
                    mixed.append(bucket.pop())
        all_messages = mixed
    else:
        all_messages = [m for _, m in prepared]

    return Dataset.from_dict({"messages": all_messages})


def load_rlvr_with_curriculum(
    path: str | Path,
    system_prompt: str | None = None,
) -> Dataset:
    """Load RLVR data with difficulty_tag column for curriculum learning."""
    return load_rlvr_dataset(path, system_prompt=system_prompt)
