"""Tests for Module 3 — Reward Composer."""

import math

import pytest

from rlvr.reward_composer import (
    W_ANSWER_LEAK,
    W_CORRECTNESS,
    W_FORMAT,
    W_LANGUAGE,
    W_LENGTH,
    W_STRUCTURAL_LEAK,
    compose_reward,
    penalty_answer_leak,
    penalty_length,
    penalty_structural_leak,
    reward_correctness,
    reward_format,
    reward_language_consistency,
)
from .fixtures.hack_patterns import CLEAN_PATTERNS, HACK_PATTERNS


def _bow_embed(text: str, dim: int = 64):
    vec = [0.0] * dim
    for tok in text.lower().split():
        vec[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def test_correctness_exact_numeric_match_true_positive():
    completion = "<think>some reasoning</think><answer>16.0</answer>"
    assert reward_correctness(completion, 16, "math") == 1.0
    assert reward_correctness(completion, 16.0, "math") == 1.0


def test_correctness_exact_numeric_match_false_positive():
    completion = "<think>some reasoning</think><answer>17</answer>"
    assert reward_correctness(completion, 16, "math") == 0.0


def test_correctness_constraint_checker_validates_full_solution_not_partial():
    gt = {"A": 1, "B": 2, "C": 3}
    partial = '<think>reasoning</think><answer>{"A": 1, "B": 2, "C": 9}</answer>'
    assert reward_correctness(partial, gt, "logic") == 0.0

    full = '<think>reasoning</think><answer>{"A": 1, "B": 2, "C": 3}</answer>'
    assert reward_correctness(full, gt, "logic") == 1.0


def test_correctness_missing_answer_tag_returns_zero():
    assert reward_correctness("<think>reasoning</think> no answer", 4, "math") == 0.0


def test_format_reward_rejects_empty_think_block():
    assert reward_format("<think></think><answer>42</answer>") == 0.0


def test_format_reward_rejects_gibberish_inside_think_tags():
    assert reward_format("<think>blah blah blah blah blah blah blah blah</think><answer>42</answer>") == 0.0


def test_format_reward_partial_scores_between_0_and_1():
    completion = "<think>short reasoning here now</think><answer>4</answer>"
    score = reward_format(completion)
    assert 0.0 < score < 1.0


def test_format_reward_rejects_missing_tags():
    assert reward_format("no tags here at all") == 0.0
    assert reward_format("<think>content</think> no answer tag") == 0.0


def test_duplicate_answer_tag_is_rejected_as_structural_leak():
    completion = (
        "<think>أحل المسألة خطوة خطوة وأتحقق من النتيجة بعناية كاملة</think>"
        "<answer>4</answer><answer>999</answer>"
    )
    assert reward_format(completion) == 0.0
    assert penalty_structural_leak(completion) == 1.0


def test_uppercase_tags_do_not_bypass_structural_leak():
    completion = (
        "one two three four five six "
        "<THINK>تفكير عربي كاف للتحقق من المسألة خطوة خطوة</THINK>"
        "<ANSWER>4</ANSWER>"
    )
    assert penalty_structural_leak(completion, max_preamble_words=5) == 1.0


def test_language_reward_penalizes_english_reasoning_in_think_block():
    completion = "<think>First I add two numbers to get the sum then verify</think><answer>4</answer>"
    assert reward_language_consistency(completion, "ar") < 0.3


def test_language_reward_does_not_penalize_math_symbols_or_numbers():
    completion = "<think>\u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 2 + 2 = 4 \u062b\u0645 \u0623\u062a\u0623\u0643\u062f</think><answer>4</answer>"
    assert reward_language_consistency(completion, "ar") > 0.8


def test_language_reward_rejects_numeric_only_reasoning():
    completion = "<think>2 + 2 = 4</think><answer>4</answer>"
    assert reward_language_consistency(completion, "ar") == 0.0


def test_answer_leak_penalty_flags_direct_answer_restatement():
    completion = "<think>the answer is definitely 42 so that is final</think><answer>42</answer>"
    result = penalty_answer_leak(completion, None, ["the answer is definitely"], threshold=0.75)
    assert result == 1.0


def test_answer_leak_penalty_does_not_flag_genuine_stepwise_reasoning():
    completion = "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u062b\u0645 \u0623\u0636\u0631\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0648\u0623\u062a\u062d\u0642\u0642 \u0645\u0646 \u0627\u0644\u0639\u0645\u0644\u064a\u0629 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629 \u0648\u0627\u0644\u0646\u0627\u062a\u062c \u0647\u0648 4</think><answer>4</answer>"
    result = penalty_answer_leak(completion, None, ["the answer is definitely", "\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a"], threshold=0.75)
    assert result == 0.0


def test_answer_leak_penalty_with_embed_fn_triggers_on_similarity():
    completion = "<think>the answer is definitely 42 so that is final</think><answer>42</answer>"
    result = penalty_answer_leak(completion, _bow_embed, ["the answer is definitely 42 so that is final"], threshold=0.75)
    assert result == 1.0


def test_structural_leak_penalty_flags_reasoning_outside_tags():
    completion = "First I add two and two to get four. <think></think><answer>4</answer>"
    assert penalty_structural_leak(completion, max_preamble_words=5) == 1.0


def test_structural_leak_penalty_allows_short_preamble():
    completion = "ok <think>real reasoning happens here with enough tokens yes</think><answer>4</answer>"
    assert penalty_structural_leak(completion, max_preamble_words=5) == 0.0


def test_length_penalty_zero_under_soft_limit():
    think = " ".join(["كلمة"] * 50)
    completion = f"<think>{think}</think><answer>4</answer>"
    assert penalty_length(completion) == 0.0


def test_length_penalty_full_over_hard_limit():
    think = " ".join(["كلمة"] * 200)
    completion = f"<think>{think}</think><answer>4</answer>"
    assert penalty_length(completion) == 1.0


def test_length_penalty_ramps_between_limits():
    think = " ".join(["كلمة"] * 120)  # midpoint of 80..160
    completion = f"<think>{think}</think><answer>4</answer>"
    assert penalty_length(completion) == pytest.approx(0.5, abs=1e-6)


def test_length_penalty_uses_full_text_when_think_unclosed():
    # Truncated mid-think — no </think>
    text = " ".join(["كلمة"] * 200)
    assert penalty_length(text) == 1.0


def test_compose_reward_formula_matches_documented_weights():
    completion = "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629</think><answer>4</answer>"
    r_correct = reward_correctness(completion, 4, "math")
    r_format = reward_format(completion)
    r_lang = reward_language_consistency(completion)
    p_leak = penalty_answer_leak(completion, None, [])
    p_struct = penalty_structural_leak(completion)
    p_length = penalty_length(completion)

    expected = (
        W_CORRECTNESS * r_correct
        + W_FORMAT * r_format
        + W_LANGUAGE * r_lang
        - W_ANSWER_LEAK * p_leak
        - W_STRUCTURAL_LEAK * p_struct
        - W_LENGTH * p_length
    )
    expected = max(0.0, min(1.0, expected))

    assert compose_reward(completion, 4, "math") == pytest.approx(expected, abs=1e-9)
    assert W_CORRECTNESS == 0.6
    assert W_FORMAT == 0.2
    assert W_LANGUAGE == 0.2
    assert W_ANSWER_LEAK == 0.5
    assert W_STRUCTURAL_LEAK == 0.3
    assert W_LENGTH == 0.15


def test_compose_reward_clamped_to_unit_interval():
    upper = "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u0645\u0639\u0627 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629</think><answer>4</answer>"
    assert compose_reward(upper, 4, "math") <= 1.0

    lower = "extra text outside yes more here <think>the answer is the answer is the answer is the answer is</think><answer>42</answer>"
    assert compose_reward(lower, 42, "math") >= 0.0
    assert compose_reward(lower, 42, "math") == 0.0


def test_reward_hacking_regression_suite():
    assert len(HACK_PATTERNS) >= 8
    for pattern in HACK_PATTERNS:
        score = compose_reward(
            pattern["completion"],
            pattern["ground_truth"],
            pattern["domain"],
        )
        assert score <= pattern["max_composite"], (
            f"Hack pattern '{pattern['name']}' scored {score}, expected <= {pattern['max_composite']}"
        )

    assert len(CLEAN_PATTERNS) >= 3
    for pattern in CLEAN_PATTERNS:
        score = compose_reward(
            pattern["completion"],
            pattern["ground_truth"],
            pattern["domain"],
        )
        assert score >= pattern["min_composite"], (
            f"Clean pattern '{pattern['name']}' scored {score}, expected >= {pattern['min_composite']}"
        )
