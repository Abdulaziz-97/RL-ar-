"""Tests for the E2H Reasoner Gaussian Curriculum Scheduler."""

import random

import pytest

from rlvr.curriculum import (
    BUCKET_NAMES,
    compute_stage_sampling_weights,
    fixed_switch_weights,
    get_curriculum_weights,
    no_curriculum_weights,
    random_mix_weights,
    sample_curriculum_bucket,
)


def test_schedule_never_assigns_exactly_zero_probability_to_any_bucket():
    total_steps = 1000
    for step in range(total_steps):
        weights = compute_stage_sampling_weights(step, total_steps)
        for bucket in BUCKET_NAMES:
            assert weights[bucket] > 0.0, (
                f"bucket {bucket} got zero weight at step {step}"
            )


@pytest.mark.parametrize("sigma", [0, -0.1, float("nan")])
def test_gaussian_schedule_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma_fraction"):
        compute_stage_sampling_weights(1, 100, sigma)


def test_schedule_weights_sum_to_one_at_every_step():
    total_steps = 1000
    for step in [0, 1, 100, 250, 350, 500, 650, 900, 999]:
        weights = compute_stage_sampling_weights(step, total_steps)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_schedule_peaks_at_correct_center_step_per_bucket():
    total_steps = 1000
    centers = {
        "trivial": 0.10 * total_steps,
        "easy": 0.35 * total_steps,
        "medium": 0.65 * total_steps,
        "hard": 0.90 * total_steps,
    }
    tolerance = 0.10 * total_steps

    for bucket, center in centers.items():
        argmax_step = 0
        best = -1.0
        for step in range(total_steps + 1):
            w = compute_stage_sampling_weights(step, total_steps)[bucket]
            if w > best:
                best = w
                argmax_step = step
        assert abs(argmax_step - center) <= tolerance, (
            f"bucket {bucket}: argmax step {argmax_step} not within "
            f"+/-{tolerance} of center {center}"
        )


def test_fixed_switch_mode_produces_hard_zero_at_boundaries():
    total_steps = 1000
    boundary_step = int(0.25 * total_steps)

    weights = fixed_switch_weights(boundary_step, total_steps)

    assert weights["trivial"] == 0.0
    assert weights["easy"] == 1.0
    zero_buckets = [b for b in BUCKET_NAMES if weights[b] == 0.0]
    assert len(zero_buckets) >= 1


def test_curriculum_schedule_type_switches_without_code_change():
    total_steps = 1000
    step = 100

    gaussian = get_curriculum_weights("gaussian", step, total_steps)
    fixed = get_curriculum_weights("fixed_switch", step, total_steps)
    mixed = get_curriculum_weights("random_mix", step, total_steps)
    none = get_curriculum_weights("none", step, total_steps)

    assert max(gaussian.values()) - min(gaussian.values()) > 1e-6
    assert sum(1 for v in fixed.values() if v == 1.0) == 1
    assert sum(fixed.values()) == pytest.approx(1.0, abs=1e-9)
    for bucket in BUCKET_NAMES:
        assert mixed[bucket] == 0.25
    assert none == {}


def test_sample_curriculum_bucket_respects_weights():
    degenerate = {"trivial": 1.0, "easy": 0.0, "medium": 0.0, "hard": 0.0}
    rng = random.Random(0)
    for _ in range(500):
        assert sample_curriculum_bucket(degenerate, rng=rng) == "trivial"

    uniform = {name: 0.25 for name in BUCKET_NAMES}
    rng = random.Random(42)
    seen = set()
    for _ in range(5000):
        seen.add(sample_curriculum_bucket(uniform, rng=rng))
    assert seen == set(BUCKET_NAMES)


def test_sample_curriculum_bucket_empty_returns_none():
    assert sample_curriculum_bucket({}) is None
    assert sample_curriculum_bucket({}, rng=random.Random(7)) is None


def test_zero_total_steps_returns_uniform():
    weights = compute_stage_sampling_weights(step=0, total_steps=0)
    for bucket in BUCKET_NAMES:
        assert weights[bucket] == 0.25
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
