"""Reward contract regression tests for V4 production baseline."""

from __future__ import annotations

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.rewards import (
    REWARD_FUNCS_ORDER,
    answer_leak_penalty_func,
    correctness_reward_func,
    diagnose_reward_weights,
    format_reward_func,
    language_reward_func,
    length_penalty_func,
)


REPO_V4 = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "qwen_4b_2x5090_v4_sota.yaml"


def _completion(text: str):
    return [[{"role": "assistant", "content": text}]]


def test_diagnose_reward_weights_v4():
    cfg = RLVRConfig.from_yaml(REPO_V4)
    report = diagnose_reward_weights(cfg.reward_weights)
    assert report["weights"]["correctness"] >= report["weights"]["format"]
    assert report["weights"]["length"] == 0.0


def test_reward_ordering_gold_beats_wrong():
    gold = "<think>خطوات</think><answer>42</answer>"
    wrong = "<think>خطوات</think><answer>7</answer>"
    malformed = "42"
    leak = "<think>الإجابة النهائية هي 42</think><answer>42</answer>"

    prompts = ["q"]
    gt = ["42"]
    domain = ["math"]
    spec = ['{"type":"integer","canonical":42}']

    gold_c = correctness_reward_func(
        prompts, _completion(gold), ground_truth_answer=gt, domain=domain, answer_spec=spec
    )[0]
    wrong_c = correctness_reward_func(
        prompts, _completion(wrong), ground_truth_answer=gt, domain=domain, answer_spec=spec
    )[0]
    assert gold_c > wrong_c

    gold_f = format_reward_func(prompts, _completion(gold))[0]
    bad_f = format_reward_func(prompts, _completion(malformed))[0]
    assert gold_f > bad_f

    leak_p = answer_leak_penalty_func(prompts, _completion(leak))[0]
    clean_p = answer_leak_penalty_func(prompts, _completion(gold))[0]
    assert leak_p <= clean_p  # more negative or equal

    ar = language_reward_func(prompts, _completion("<think>هذا نص عربي طويل بما يكفي</think><answer>1</answer>"))[0]
    en = language_reward_func(prompts, _completion("<think>this is entirely english reasoning text here</think><answer>1</answer>"))[0]
    assert ar >= en


def test_length_penalty_configurable_via_kwargs():
    long_think = "<think>" + ("كلمة " * 200) + "</think><answer>1</answer>"
    default = length_penalty_func(["q"], _completion(long_think))[0]
    calibrated = length_penalty_func(
        ["q"], _completion(long_think), soft_limit=500, hard_limit=1000
    )[0]
    # With much larger limits, penalty should be weaker (closer to 0).
    assert calibrated >= default


def test_reward_funcs_order_stable():
    assert REWARD_FUNCS_ORDER[0] == "correctness"
    assert len(REWARD_FUNCS_ORDER) == 6
