import math

import pytest

from rlvr.entropy_guard import (
    apply_entropy_guard,
    clip_cov_mask,
    compute_batch_entropy,
    compute_token_covariance,
    kl_cov_penalty,
)


def test_covariance_computation_matches_hand_derived_example():
    log_probs = [-0.5, -0.2, -0.8, -0.3, -0.6]
    advantages = [1.0, -0.5, 0.3, -0.2, 0.4]
    result = compute_token_covariance(log_probs, advantages)
    expected = [-0.0032, -0.0392, -0.0064, -0.0144, -0.0048]
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-9
    assert abs(sum(result) - (-0.068)) < 1e-9


def test_clip_cov_mask_selects_exactly_top_fraction():
    covariances = [float(i) for i in range(100)]
    mask = clip_cov_mask(covariances, clip_ratio=0.02)
    assert sum(mask) == 2
    true_indices = [i for i, m in enumerate(mask) if m]
    assert true_indices == [98, 99]
    small = [0.1, 0.2, 0.3, 0.4, 0.5]
    small_mask = clip_cov_mask(small, clip_ratio=0.02)
    assert sum(small_mask) == 1


def test_clip_cov_excludes_masked_tokens_from_gradient():
    loss_terms = [1.0, 2.0, 3.0, 4.0, 5.0]
    mask = [True, False, False, False, False]
    result = apply_entropy_guard(loss_terms, mask, "clip_cov")
    assert result == [0.0, 2.0, 3.0, 4.0, 5.0]


def test_kl_cov_penalty_only_applied_to_top_k_tokens():
    covariances = [float(i) for i in range(100)]
    penalties = kl_cov_penalty(covariances, kl_coef=1.0, top_k_ratio=0.02)
    nonzero = [p for p in penalties if p != 0.0]
    assert len(nonzero) == 2
    assert penalties[98] == 98.0 / 99.0
    assert penalties[99] == 1.0


def test_kl_cov_penalty_magnitude_scales_with_kl_coef():
    covariances = [float(i) for i in range(100)]
    p1 = kl_cov_penalty(covariances, kl_coef=1.0, top_k_ratio=0.02)
    p2 = kl_cov_penalty(covariances, kl_coef=2.0, top_k_ratio=0.02)
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert abs(b - 2.0 * a) < 1e-9


def test_entropy_guard_mode_flag_switches_strategy_without_code_change():
    loss_terms = [1.0, 2.0, 3.0, 4.0, 5.0]
    mask = [True, False, False, False, False]
    clip_result = apply_entropy_guard(loss_terms, mask, "clip_cov")
    penalties = [0.5, 0.0, 0.0, 0.0, 0.0]
    kl_result = apply_entropy_guard(loss_terms, penalties, "kl_cov")
    assert clip_result == [0.0, 2.0, 3.0, 4.0, 5.0]
    assert kl_result == [1.5, 2.0, 3.0, 4.0, 5.0]
    assert clip_result != kl_result


def test_batch_entropy_computation_matches_standard_definition():
    ln_half = math.log(0.5)
    single = compute_batch_entropy([[ln_half, ln_half]])
    assert abs(single - math.log(2)) < 1e-6
    two = compute_batch_entropy([[ln_half, ln_half], [ln_half, ln_half]])
    assert abs(two - math.log(2)) < 1e-6


def test_entropy_guard_does_not_crash_on_uniform_covariance_batch():
    covariances = [0.05, 0.05, 0.05, 0.05, 0.05]
    mask = clip_cov_mask(covariances, clip_ratio=0.02)
    assert sum(mask) >= 1
    penalties = kl_cov_penalty(covariances, kl_coef=1.0, top_k_ratio=0.02)
    assert len(penalties) == 5
    assert sum(1 for p in penalties if p != 0.0) >= 1


def test_apply_entropy_guard_length_mismatch_raises():
    with pytest.raises(ValueError):
        apply_entropy_guard([1.0, 2.0], [True], "clip_cov")
    with pytest.raises(ValueError):
        apply_entropy_guard([1.0, 2.0], [0.1], "kl_cov")


def test_compute_batch_entropy_empty_returns_zero():
    assert compute_batch_entropy([]) == 0.0
    assert compute_batch_entropy([[], []]) == 0.0
