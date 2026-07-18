"""Tests for the data loading module with real Arabic Reasoning schema."""

from datasets import Dataset

from rlvr_pipeline.data import (
    DOMAIN_MAP,
    _derive_difficulty,
    _transform_coldstart_response,
    load_rlvr_dataset,
    load_cold_start_sft_dataset,
)


def test_load_rlvr_dataset_math(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl, system_prompt="You are helpful.")
    assert isinstance(ds, Dataset)
    assert len(ds) == 8
    assert "prompt" in ds.column_names
    assert "ground_truth_answer" in ds.column_names
    assert "domain" in ds.column_names
    assert "difficulty_tag" in ds.column_names
    assert "sample_id" in ds.column_names


def test_load_rlvr_dataset_conversational_format(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl, system_prompt="sys")
    first = ds[0]
    assert isinstance(first["prompt"], list)
    assert first["prompt"][0]["role"] == "system"
    assert first["prompt"][0]["content"] == "sys"
    assert first["prompt"][1]["role"] == "user"


def test_load_rlvr_dataset_no_system_prompt(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl, system_prompt=None)
    first = ds[0]
    assert first["prompt"][0]["role"] == "user"


def test_load_rlvr_dataset_domain_mapping(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl)
    for row in ds:
        assert row["domain"] == "math"


def test_load_rlvr_dataset_logic(logic_jsonl):
    ds = load_rlvr_dataset(logic_jsonl)
    assert len(ds) == 4
    for row in ds:
        assert row["domain"] == "logic"
        assert isinstance(row["ground_truth_answer"], str)


def test_load_rlvr_dataset_extracts_gt_from_metadata(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl)
    for i, row in enumerate(ds):
        assert row["ground_truth_answer"] == str((i + 1) + (i + 1))


def test_load_rlvr_dataset_filters_empty_gt(mixed_domain_jsonl):
    ds = load_rlvr_dataset(mixed_domain_jsonl)
    for row in ds:
        assert row["ground_truth_answer"] != ""
        assert row["ground_truth_answer"] is not None


def test_load_rlvr_dataset_has_difficulty_tags(math_jsonl):
    ds = load_rlvr_dataset(math_jsonl)
    tags = set(ds["difficulty_tag"])
    assert tags.issubset({"trivial", "easy", "medium", "hard"})


def test_derive_difficulty_from_num_steps():
    assert _derive_difficulty({"num_steps": 1}) == "trivial"
    assert _derive_difficulty({"num_steps": 3}) == "easy"
    assert _derive_difficulty({"num_steps": 5}) == "medium"
    assert _derive_difficulty({"num_steps": 10}) == "hard"


def test_derive_difficulty_from_grade_level():
    assert _derive_difficulty({"grade_level": "grade_2"}) == "easy"
    assert _derive_difficulty({"grade_level": "grade_5"}) == "medium"
    assert _derive_difficulty({"grade_level": "grade_8"}) == "hard"


def test_derive_difficulty_default():
    assert _derive_difficulty({}) == "medium"


def test_domain_mapping():
    assert DOMAIN_MAP["gsm8k"] == "math"
    assert DOMAIN_MAP["math"] == "math"
    assert DOMAIN_MAP["math_comp"] == "math"
    assert DOMAIN_MAP["logic"] == "logic"
    assert DOMAIN_MAP.get("mmlu", "math") == "math"


def test_transform_coldstart_response_adds_answer_tags():
    resp = "<think>\nStep 1: 2+2=4\n</think>\n\n#### 4"
    result = _transform_coldstart_response(resp)
    assert "<answer>4</answer>" in result
    assert "####" not in result


def test_transform_coldstart_response_preserves_existing_tags():
    resp = "<think>reasoning</think><answer>4</answer>"
    result = _transform_coldstart_response(resp)
    assert result == resp


def test_load_coldstart_sft_dataset(coldstart_jsonl):
    ds = load_cold_start_sft_dataset(coldstart_jsonl, system_prompt="sys")
    assert len(ds) == 8
    assert "messages" in ds.column_names
    first = ds[0]["messages"]
    assert len(first) == 3
    assert first[0]["role"] == "system"
    assert first[1]["role"] == "user"
    assert first[2]["role"] == "assistant"
    assert "<answer>" in first[2]["content"]
    assert "####" not in first[2]["content"]
