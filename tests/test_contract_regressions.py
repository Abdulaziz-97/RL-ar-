"""Regression tests for strict answer and response contracts."""

import pytest

from rlvr_contracts.answer_spec import AnswerSpec, AnswerSpecError, parse_answer_spec
from rlvr_contracts.leak import length_penalty_score
from rlvr_contracts.response import validate_response_format


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_decimal_approx_rejects_non_finite_values(value):
    with pytest.raises(AnswerSpecError):
        parse_answer_spec(
            {"type": "decimal_approx", "canonical": value, "tolerance": 1e-6}
        )


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf"), 1.1])
def test_decimal_approx_rejects_invalid_tolerance(tolerance):
    with pytest.raises(AnswerSpecError):
        parse_answer_spec(
            {"type": "decimal_approx", "canonical": "1.5", "tolerance": tolerance}
        )


def test_unbroken_character_padding_is_rejected():
    think = "س" * 10_000
    completion = f"<think>{think}</think><answer>4</answer>"
    assert validate_response_format(completion) == 0.0
    assert length_penalty_score(completion) == 1.0


def test_preconstructed_answer_spec_is_revalidated():
    with pytest.raises(AnswerSpecError):
        parse_answer_spec(
            AnswerSpec(type="decimal_approx", canonical="nan", tolerance=-1)
        )


def test_logic_structured_ground_truth_must_match_canonical():
    with pytest.raises(AnswerSpecError, match="conflicts"):
        parse_answer_spec(
            {
                "type": "logic_json",
                "canonical": {"x": 1},
                "ground_truth_structured": {"x": 2},
            }
        )


@pytest.mark.parametrize("value", ['{"x":NaN}', {"x": float("inf")}])
def test_logic_json_rejects_nonstandard_numbers(value):
    with pytest.raises(AnswerSpecError):
        parse_answer_spec({"type": "logic_json", "canonical": value})
