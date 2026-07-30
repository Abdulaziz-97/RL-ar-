"""Tests for Module 4 — Difficulty Labeler."""

import hashlib
import random

import pytest

from rlvr.difficulty_labeler import (
    QUARTILE_LABELS,
    assign_difficulty_quartiles,
    compute_error_rate,
)


class ProbabilisticModel:
    def __init__(self, correct_probability: float, seed: int = 0):
        self._p = correct_probability
        self._rng = random.Random(seed)
        self.calls = 0

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        self.calls += 1
        if self._rng.random() < self._p:
            return f"<think>reasoning</think><answer>42</answer>"
        return f"<think>reasoning</think><answer>0</answer>"


class DeterministicModel:
    def __init__(self, seed: int = 12345):
        self._seed = seed
        self.calls = 0

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        self.calls += 1
        h = int(hashlib.md5(f"{self._seed}:{prompt}:{self.calls}".encode()).hexdigest(), 16)
        if h % 10 < 7:
            return f"<think>reasoning</think><answer>42</answer>"
        return f"<think>reasoning</think><answer>0</answer>"


def _make_sample(domain: str = "math", ground_truth=42):
    return {
        "id": "s1",
        "domain": domain,
        "difficulty_tag": "easy",
        "prompt": "What is 2+2?",
        "ground_truth_answer": ground_truth,
    }


def test_error_rate_is_one_minus_correct_fraction():
    model = ProbabilisticModel(correct_probability=1.0, seed=1)
    sample = _make_sample()
    rate = compute_error_rate(model, sample, n_attempts=20)
    assert rate == pytest.approx(0.0, abs=1e-9)

    model = ProbabilisticModel(correct_probability=0.0, seed=2)
    rate = compute_error_rate(model, sample, n_attempts=20)
    assert rate == pytest.approx(1.0, abs=1e-9)

    model = ProbabilisticModel(correct_probability=0.5, seed=3)
    rate = compute_error_rate(model, sample, n_attempts=20)
    assert 0.0 <= rate <= 1.0


def test_quartile_assignment_produces_four_roughly_equal_buckets():
    n = 200
    samples = [{**_make_sample(), "id": f"s{i}"} for i in range(n)]
    error_rates = [i / n for i in range(n)]

    refined = assign_difficulty_quartiles(samples, error_rates)

    buckets = {label: 0 for label in QUARTILE_LABELS}
    for s in refined:
        buckets[s["difficulty_tag"]] += 1

    for label in QUARTILE_LABELS:
        assert buckets[label] > 0
    assert sum(buckets.values()) == n
    assert max(buckets.values()) - min(buckets.values()) <= 1


def test_all_trivial_or_all_hard_dataset_still_splits_without_crash():
    n = 20
    samples = [{**_make_sample(), "id": f"s{i}"} for i in range(n)]
    error_rates = [0.0] * n

    refined = assign_difficulty_quartiles(samples, error_rates)
    buckets = {label: 0 for label in QUARTILE_LABELS}
    for s in refined:
        buckets[s["difficulty_tag"]] += 1
    assert sum(buckets.values()) == n

    error_rates = [1.0] * n
    refined = assign_difficulty_quartiles(samples, error_rates)
    assert len(refined) == n


def test_error_rate_stable_across_repeated_runs_at_fixed_n():
    sample = _make_sample()
    rates = []
    for run in range(5):
        model = DeterministicModel(seed=999)
        rate = compute_error_rate(model, sample, n_attempts=20)
        rates.append(rate)
    spread = max(rates) - min(rates)
    assert spread <= 0.1


def test_empty_samples_returns_empty():
    assert assign_difficulty_quartiles([], []) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        assign_difficulty_quartiles([_make_sample()], [0.5, 0.6])


def test_quartile_ordering_matches_error_rate():
    n = 8
    samples = [{**_make_sample(), "id": f"s{i}"} for i in range(n)]
    error_rates = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]

    refined = assign_difficulty_quartiles(samples, error_rates)
    by_id = {s["id"]: s["difficulty_tag"] for s in refined}

    rank = {label: i for i, label in enumerate(QUARTILE_LABELS)}
    pairs = sorted(zip(error_rates, [s["id"] for s in samples]))
    for i in range(len(pairs) - 1):
        low_rate, low_id = pairs[i]
        high_rate, high_id = pairs[i + 1]
        assert rank[by_id[low_id]] <= rank[by_id[high_id]]
