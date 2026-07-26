"""Tests for benchmark evaluation logic."""

import json
from pathlib import Path
from rlvr_pipeline.rewards import reward_correctness, reward_format


def test_reward_scoring_on_english_completions():
    completions = [
        "Thinking step-by-step:\n5 + 7 = 12.\n<answer>12</answer>",
        "Wrong answer:\n<answer>15</answer>",
        "Malformed completion without tags 12",
    ]
    answer_specs = [
        {"type": "integer", "canonical": "12"},
        {"type": "integer", "canonical": "12"},
        {"type": "integer", "canonical": "12"},
    ]

    correctness = reward_correctness(completions, answer_specs)
    assert correctness[0] == 1.0
    assert correctness[1] == 0.0
    assert correctness[2] == 0.0

    formats = reward_format(completions)
    assert formats[0] == 1.0
    assert formats[1] == 1.0
    assert formats[2] == 0.0
