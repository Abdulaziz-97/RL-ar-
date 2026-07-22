import json

import pytest

from rlvr.dataset_loader import (
    SchemaValidationError,
    validate_cold_start_sample,
    validate_rlvr_sample,
    load_cold_start_dataset,
    load_rlvr_dataset,
    find_near_duplicate_prompts,
)


def _valid_cold_start(**overrides) -> dict:
    base = {
        "id": "cs-1",
        "domain": "math",
        "difficulty": "easy",
        "prompt": "ما هو ناتج جمع واحد زائد واحد؟",
        "response": "<think>نجمع واحد مع واحد فيساوي اثنين</think><answer>2</answer>",
    }
    base.update(overrides)
    return base


def _valid_rlvr_math(**overrides) -> dict:
    base = {
        "id": "rlvr-m-1",
        "domain": "math",
        "difficulty_tag": "easy",
        "prompt": "احسب ناتج 2 + 2.",
        "ground_truth_answer": "4",
    }
    base.update(overrides)
    return base


def test_validate_rlvr_rejects_requested_domain_mismatch():
    with pytest.raises(SchemaValidationError, match="expected domain"):
        validate_rlvr_sample(_valid_rlvr_math(), "logic")


def _valid_rlvr_logic(**overrides) -> dict:
    base = {
        "id": "rlvr-l-1",
        "domain": "logic",
        "difficulty_tag": "medium",
        "prompt": "إذا كان كل القطط حيوانات وكل الحيوانات تتنفس فهل القطط تتنفس؟",
        "ground_truth_answer": "نعم",
        "puzzle_type": "syllogism",
    }
    base.update(overrides)
    return base


def test_rejects_sample_missing_think_tags():
    missing_think = _valid_cold_start(response="<answer>2</answer>")
    with pytest.raises(SchemaValidationError):
        validate_cold_start_sample(missing_think)

    missing_answer = _valid_cold_start(response="<think>بعض التفكير</think>")
    with pytest.raises(SchemaValidationError):
        validate_cold_start_sample(missing_answer)


def test_rejects_duplicate_ids_across_coldstart_and_rlvr():
    near_cs = validate_cold_start_sample(
        _valid_cold_start(id="cs-1", prompt="Compute  the  sum  of  two  and  three")
    )
    near_rlvr = validate_rlvr_sample(
        _valid_rlvr_math(id="rl-1", prompt="Compute the sum of two and three"), "math"
    )
    flagged = find_near_duplicate_prompts([near_cs], [near_rlvr])
    assert ("cs-1", "rl-1") in flagged

    exact_cs = validate_cold_start_sample(
        _valid_cold_start(id="cs-2", prompt="What is the square root of sixteen")
    )
    exact_rlvr = validate_rlvr_sample(
        _valid_rlvr_math(id="rl-2", prompt="What is the square root of sixteen"), "math"
    )
    flagged = find_near_duplicate_prompts([exact_cs], [exact_rlvr])
    assert ("cs-2", "rl-2") in flagged


def test_accepts_valid_minimal_sample():
    cold = validate_cold_start_sample(_valid_cold_start())
    assert cold["id"] == "cs-1"
    assert set(cold.keys()) == {"id", "domain", "difficulty", "prompt", "response"}

    math_sample = validate_rlvr_sample(_valid_rlvr_math(), "math")
    assert math_sample["domain"] == "math"
    assert "puzzle_type" not in math_sample

    logic_sample = validate_rlvr_sample(_valid_rlvr_logic(), "logic")
    assert logic_sample["domain"] == "logic"
    assert logic_sample.get("puzzle_type") == "syllogism"


def test_domain_field_only_allows_math_or_logic():
    sample = _valid_rlvr_math(domain="history")
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(sample, "math")


def test_logic_samples_require_puzzle_type_field():
    missing_puzzle = _valid_rlvr_logic()
    del missing_puzzle["puzzle_type"]
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(missing_puzzle, "logic")

    blank_puzzle = _valid_rlvr_logic(puzzle_type="   ")
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(blank_puzzle, "logic")


def test_math_samples_are_valid_without_puzzle_type_field():
    sample = _valid_rlvr_math()
    assert "puzzle_type" not in sample
    result = validate_rlvr_sample(sample, "math")
    assert "puzzle_type" not in result
    assert result["ground_truth_answer"] == "4"


def test_load_cold_start_dataset_reads_json_array(tmp_path):
    samples = [_valid_cold_start(id="cs-1"), _valid_cold_start(id="cs-2")]
    path = tmp_path / "cold.json"
    path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    loaded = load_cold_start_dataset(str(path))
    assert len(loaded) == 2
    assert loaded[0]["id"] == "cs-1"
    assert loaded[1]["id"] == "cs-2"
    assert loaded[0]["response"].startswith("<think>")


def test_load_rlvr_dataset_reads_jsonl(tmp_path):
    samples = [_valid_rlvr_math(id="m-1"), _valid_rlvr_math(id="m-2")]
    path = tmp_path / "rlvr.jsonl"
    path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
        encoding="utf-8",
    )
    loaded = load_rlvr_dataset(str(path), "math")
    assert len(loaded) == 2
    assert loaded[0]["id"] == "m-1"
    assert loaded[1]["id"] == "m-2"


def test_duplicate_ids_within_dataset_raises(tmp_path):
    cold_samples = [_valid_cold_start(id="dup"), _valid_cold_start(id="dup")]
    cold_path = tmp_path / "cold_dup.json"
    cold_path.write_text(json.dumps(cold_samples, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_cold_start_dataset(str(cold_path))

    rlvr_samples = [_valid_rlvr_math(id="dup"), _valid_rlvr_math(id="dup")]
    rlvr_path = tmp_path / "rlvr_dup.jsonl"
    rlvr_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in rlvr_samples),
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationError):
        load_rlvr_dataset(str(rlvr_path), "math")


def test_missing_required_fields_raises():
    missing_id = _valid_cold_start()
    del missing_id["id"]
    with pytest.raises(SchemaValidationError):
        validate_cold_start_sample(missing_id)

    missing_prompt = _valid_cold_start()
    del missing_prompt["prompt"]
    with pytest.raises(SchemaValidationError):
        validate_cold_start_sample(missing_prompt)

    rlvr_missing_id = _valid_rlvr_math()
    del rlvr_missing_id["id"]
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(rlvr_missing_id, "math")

    rlvr_missing_prompt = _valid_rlvr_math()
    del rlvr_missing_prompt["prompt"]
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(rlvr_missing_prompt, "math")

    rlvr_missing_gt = _valid_rlvr_math()
    del rlvr_missing_gt["ground_truth_answer"]
    with pytest.raises(SchemaValidationError):
        validate_rlvr_sample(rlvr_missing_gt, "math")


def test_near_duplicate_threshold_respected():
    high_cs = validate_cold_start_sample(
        _valid_cold_start(id="cs-hi", prompt="the quick brown fox jumps over the lazy dog")
    )
    high_rl = validate_rlvr_sample(
        _valid_rlvr_math(id="rl-hi", prompt="the quick brown fox jumps over the lazy dog"), "math"
    )
    flagged = find_near_duplicate_prompts([high_cs], [high_rl], threshold=0.8)
    assert ("cs-hi", "rl-hi") in flagged

    low_cs = validate_cold_start_sample(
        _valid_cold_start(id="cs-lo", prompt="alpha beta gamma delta epsilon zeta eta")
    )
    low_rl = validate_rlvr_sample(
        _valid_rlvr_math(id="rl-lo", prompt="one two three four five six seven"), "math"
    )
    flagged = find_near_duplicate_prompts([low_cs], [low_rl], threshold=0.8)
    assert ("cs-lo", "rl-lo") not in flagged
