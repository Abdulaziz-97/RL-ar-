"""Tests for Module 5 — Advantage Calculator."""

import math
import random

from rlvr.advantage_calculator import compute_advantages, flag_zero_variance_group


def test_advantage_formula_matches_grpo_definition():
    rewards = [1.0, 2.0, 3.0, 4.0]
    result = compute_advantages(rewards, eps=1e-4)

    mean = 2.5
    std = math.sqrt(
        ((1.0 - 2.5) ** 2 + (2.0 - 2.5) ** 2 + (3.0 - 2.5) ** 2 + (4.0 - 2.5) ** 2) / 4
    )
    denom = std + 1e-4
    expected = [(r - mean) / denom for r in rewards]

    assert len(result) == len(expected)
    for a, e in zip(result, expected):
        assert abs(a - e) < 1e-6


def test_zero_std_group_does_not_produce_nan_or_inf():
    rewards = [0.5, 0.5, 0.5, 0.5]
    result = compute_advantages(rewards)

    assert all(r == 0.0 for r in result)
    assert not any(math.isnan(r) or math.isinf(r) for r in result)


def test_flag_zero_variance_correctly_identifies_all_same_reward():
    assert flag_zero_variance_group([3, 3, 3]) is True
    assert flag_zero_variance_group([3, 4, 3]) is False


def test_mixed_reward_group_produces_expected_sign():
    rewards = [0.1, 0.5, 0.9]
    result = compute_advantages(rewards)

    assert result[-1] > 0
    assert result[0] < 0


def test_all_zero_rewards_returns_zeros_not_nan():
    result = compute_advantages([0, 0, 0, 0])

    assert result == [0.0, 0.0, 0.0, 0.0]


def test_single_outlier_edge_case():
    rewards = [0, 0, 0, 1.0]
    result = compute_advantages(rewards)

    assert not any(math.isnan(r) or math.isinf(r) for r in result)
    assert result[-1] > 0
    assert all(r < 0 for r in result[:-1])


def test_empty_list_returns_empty():
    assert compute_advantages([]) == []


def test_randomized_no_nan_across_many_vectors():
    rng = random.Random(20240101)

    for _ in range(1000):
        n = rng.randint(1, 20)
        choice = rng.randint(0, 2)

        if choice == 0:
            rewards = [0.0] * n
        elif choice == 1:
            rewards = [1.0] * n
        else:
            rewards = [rng.random() for _ in range(n)]
            if n > 1 and rng.random() < 0.5:
                rewards[rng.randrange(n)] = 5.0  # inject a single outlier

        result = compute_advantages(rewards)
        for r in result:
            assert not math.isnan(r), f"NaN produced for rewards={rewards}"
            assert not math.isinf(r), f"Inf produced for rewards={rewards}"
