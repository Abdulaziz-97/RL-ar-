"""Tests for Failure Mining (RL-ZVP + POPO)."""

import math

from rlvr.failure_mining import (
    ReplayBuffer,
    ReplayEntry,
    decoupled_importance_ratio,
    is_effective_group,
    is_zero_variance_group,
    maybe_push_to_buffer,
    replace_ineffective_group,
    score_zero_variance_group,
)


def _entry(prompt_id: str, rewards: list[float], step: int = 1) -> ReplayEntry:
    return ReplayEntry(
        prompt_id=prompt_id,
        completions=["c1", "c2"],
        rewards=rewards,
        old_policy_log_probs=[],
        step_generated=step,
    )


def test_zero_variance_detects_all_identical_rewards():
    assert is_zero_variance_group([0.5, 0.5, 0.5]) is True
    assert is_zero_variance_group([0.5, 0.6, 0.5]) is False
    assert is_zero_variance_group([0.0, 0.0, 0.0]) is True
    assert is_zero_variance_group([1.0, 1.0, 1.0]) is True


def test_zero_variance_all_wrong_penalizes_high_confidence_tokens_more():
    completions = ["c1"]
    rewards = [0.0, 0.0, 0.0]
    token_confidences = [[0.9, 0.3]]
    signals = score_zero_variance_group(completions, rewards, token_confidences)

    sig = signals[0]
    assert sig == [-0.9, -0.3]
    assert sig[0] < sig[1]


def test_zero_variance_all_correct_rewards_low_confidence_tokens_more():
    completions = ["c1"]
    rewards = [1.0, 1.0, 1.0]
    token_confidences = [[0.9, 0.3]]
    signals = score_zero_variance_group(completions, rewards, token_confidences)

    sig = signals[0]
    assert abs(sig[0] - 0.1) < 1e-9
    assert abs(sig[1] - 0.7) < 1e-9
    assert sig[1] > sig[0]


def test_direct_scoring_produces_nonzero_gradient_signal():
    completions = ["c1", "c2"]
    rewards = [0.0, 0.0]
    token_confidences = [[0.5, 0.7], [0.3, 0.8]]
    signals = score_zero_variance_group(completions, rewards, token_confidences)

    total = sum(abs(s) for comp in signals for s in comp)
    assert total > 0


def test_direct_scoring_does_not_run_on_mixed_variance_groups():
    assert is_zero_variance_group([0.0, 0.5, 1.0]) is False


def test_effective_group_includes_partial_success_not_just_failures():
    assert is_effective_group([0.0, 0.5, 1.0]) is True
    assert is_effective_group([0.0, 0.0, 0.0]) is False
    assert is_effective_group([1.0, 1.0, 1.0]) is False


def test_buffer_eviction_is_fifo_not_random():
    buffer = ReplayBuffer(max_size=3)
    for i in range(1, 6):
        buffer.push(_entry(f"p{i}", [float(i)], step=i))

    remaining = list(buffer.buffer)
    assert len(remaining) == 3
    assert [e.prompt_id for e in remaining] == ["p3", "p4", "p5"]


def test_buffer_sample_returns_none_when_empty():
    buffer = ReplayBuffer()
    assert buffer.sample() is None


def test_replay_never_exceeds_buffer_max_size():
    buffer = ReplayBuffer(max_size=10)
    for i in range(100):
        buffer.push(_entry(f"p{i}", [float(i)], step=i))

    assert len(buffer) == 10


def test_maybe_push_only_pushes_effective_groups():
    buffer = ReplayBuffer(max_size=512)

    maybe_push_to_buffer(buffer, _entry("eff", [0.0, 0.5, 1.0]))
    assert len(buffer) == 1

    maybe_push_to_buffer(buffer, _entry("ineff", [0.0, 0.0, 0.0]))
    assert len(buffer) == 1


def test_replace_ineffective_group_returns_entry_when_buffer_nonempty():
    buffer = ReplayBuffer(max_size=512)
    buffer.push(_entry("e1", [0.0, 0.5, 1.0]))

    result = replace_ineffective_group(buffer, [0.0, 0.0, 0.0])
    assert result is not None
    assert result.prompt_id == "e1"


def test_replace_ineffective_group_returns_none_for_effective_group():
    buffer = ReplayBuffer(max_size=512)
    buffer.push(_entry("e1", [0.0, 0.5, 1.0]))

    assert replace_ineffective_group(buffer, [0.0, 0.5, 1.0]) is None


def test_decoupled_ratio_clip_term_only_uses_recent_policy():
    current = [math.log(0.6), math.log(0.5)]
    recent = [math.log(0.5), math.log(0.4)]
    old_a = [math.log(0.4), math.log(0.3)]
    old_b = [math.log(0.2), math.log(0.1)]

    clip_a, _ = decoupled_importance_ratio(current, recent, old_a)
    clip_b, _ = decoupled_importance_ratio(current, recent, old_b)

    assert clip_a == clip_b


def test_decoupled_ratio_correction_term_uses_true_origin_policy():
    current = [math.log(0.6), math.log(0.5)]
    recent = [math.log(0.5), math.log(0.4)]
    old_a = [math.log(0.4), math.log(0.3)]
    old_b = [math.log(0.2), math.log(0.1)]

    _, corr_a = decoupled_importance_ratio(current, recent, old_a)
    _, corr_b = decoupled_importance_ratio(current, recent, old_b)

    assert corr_a != corr_b
    expected_corr_a = [math.exp(r - o) for r, o in zip(recent, old_a)]
    for got, exp in zip(corr_a, expected_corr_a):
        assert abs(got - exp) < 1e-9


def test_decoupled_ratio_matches_hand_derived_toy_example():
    current = [math.log(0.6), math.log(0.5)]
    recent = [math.log(0.5), math.log(0.4)]
    old = [math.log(0.4), math.log(0.3)]

    clip_ratios, correction_ratios = decoupled_importance_ratio(current, recent, old)

    expected_clip = [1.2, 1.25]
    expected_corr = [1.25, 0.4 / 0.3]

    for got, exp in zip(clip_ratios, expected_clip):
        assert abs(got - exp) < 1e-6
    for got, exp in zip(correction_ratios, expected_corr):
        assert abs(got - exp) < 1e-6


def test_decoupled_ratio_length_mismatch_handling():
    current = [math.log(0.6), math.log(0.5), math.log(0.4)]
    recent = [math.log(0.5), math.log(0.4)]
    old = [math.log(0.4)]

    clip_ratios, correction_ratios = decoupled_importance_ratio(current, recent, old)

    assert len(clip_ratios) == 2
    assert len(correction_ratios) == 1
