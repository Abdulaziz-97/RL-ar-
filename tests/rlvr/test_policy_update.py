import math

import pytest
import torch

from rlvr.entropy_guard import clip_cov_mask, compute_token_covariance
from rlvr.policy_update import grpo_loss, gspo_loss, kl_penalty_term


def test_grpo_loss_matches_hand_derived_value_on_toy_example():
    ratios = torch.tensor([1.0, 1.0, 1.0])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    lp = torch.tensor([math.log(0.5), math.log(0.5), math.log(0.5)])
    loss = grpo_loss(ratios, advantages, lp, lp)
    assert abs(loss.item() - (-0.16666666666666666)) < 1e-5

    ratios2 = torch.tensor([2.0, 0.5])
    advantages2 = torch.tensor([1.0, 1.0])
    lp2 = torch.tensor([math.log(0.5), math.log(0.5)])
    loss2 = grpo_loss(ratios2, advantages2, lp2, lp2)
    assert abs(loss2.item() - (-0.85)) < 1e-5

    ratios3 = torch.tensor([1.0, 1.0])
    advantages3 = torch.tensor([1.0, 1.0])
    ref_lp = torch.tensor([math.log(0.5), math.log(0.5)])
    policy_lp = torch.tensor([math.log(0.25), math.log(0.75)])
    loss3 = grpo_loss(ratios3, advantages3, ref_lp, policy_lp)
    assert abs(loss3.item() - (-0.9971231792754822)) < 1e-5


def test_clipping_bounds_are_respected():
    ratios = torch.tensor([2.0])
    advantages = torch.tensor([1.0])
    lp = torch.tensor([math.log(0.5)])
    loss = grpo_loss(ratios, advantages, lp, lp, clip_eps=0.2, kl_coef=0.0)
    assert abs(loss.item() - (-1.2)) < 1e-6
    loss_unclipped = grpo_loss(ratios, advantages, lp, lp, clip_eps=10.0, kl_coef=0.0)
    assert abs(loss_unclipped.item() - (-2.0)) < 1e-6


def test_gspo_uses_sequence_level_ratio_not_token_level():
    advantages = torch.tensor([1.0, -1.0, 0.5])
    lp = torch.tensor([math.log(0.5), math.log(0.5), math.log(0.5)])

    g1 = gspo_loss(torch.tensor(1.0), advantages, lp, lp)
    r1 = grpo_loss(torch.tensor([1.0, 1.0, 1.0]), advantages, lp, lp)
    assert torch.allclose(g1, r1)

    g2 = gspo_loss(torch.tensor(1.1), advantages, lp, lp)
    r2 = grpo_loss(torch.tensor([1.1, 1.1, 1.1]), advantages, lp, lp)
    assert torch.allclose(g2, r2)

    g3 = gspo_loss(torch.tensor(1.0), advantages, lp, lp)
    r3 = grpo_loss(torch.tensor([0.9, 1.0, 1.1]), advantages, lp, lp)
    assert not torch.allclose(g3, r3)


def test_grpo_kl_penalty_term_added_correctly():
    ratios = torch.tensor([1.0, 1.0])
    advantages = torch.tensor([1.0, -1.0])
    ref_lp = torch.tensor([math.log(0.5), math.log(0.5)])
    policy_lp = torch.tensor([math.log(0.25), math.log(0.75)])
    clip_eps = 0.2
    kl_coef = 0.04

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = -torch.min(surr1, surr2).mean()
    kl = (torch.exp(ref_lp) * (ref_lp - policy_lp)).mean()
    expected = policy_obj + kl_coef * kl

    actual = grpo_loss(ratios, advantages, ref_lp, policy_lp, clip_eps, kl_coef)
    assert torch.allclose(actual, expected)
    assert kl.item() != 0.0


def test_gspo_kl_penalty_term_added_correctly():
    sequence_ratios = torch.tensor(1.0)
    advantages = torch.tensor([1.0, -1.0])
    ref_lp = torch.tensor([math.log(0.5), math.log(0.5)])
    policy_lp = torch.tensor([math.log(0.25), math.log(0.75)])
    clip_eps = 0.2
    kl_coef = 0.04

    surr1 = sequence_ratios * advantages
    surr2 = torch.clamp(sequence_ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = -torch.min(surr1, surr2).mean()
    kl = (torch.exp(ref_lp) * (ref_lp - policy_lp)).mean()
    expected = policy_obj + kl_coef * kl

    actual = gspo_loss(sequence_ratios, advantages, ref_lp, policy_lp, clip_eps, kl_coef)
    assert torch.allclose(actual, expected)
    assert kl.item() != 0.0

    same_kl = kl_penalty_term(ref_lp, policy_lp)
    assert torch.allclose(kl, same_kl)


def test_policy_update_consumes_entropy_guarded_loss_terms():
    ratios = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    advantages_list = [1.0, -0.5, 0.3, -0.2, 0.4]
    log_probs_list = [-0.5, -0.2, -0.8, -0.3, -0.6]
    ref_lp = torch.tensor([math.log(0.5)] * 5)
    policy_lp = ref_lp
    clip_eps = 0.2
    kl_coef = 0.04

    covariances = compute_token_covariance(log_probs_list, advantages_list)
    mask = clip_cov_mask(covariances, clip_ratio=0.02)
    advantages_masked = [a if not m else 0.0 for a, m in zip(advantages_list, mask)]

    loss = grpo_loss(ratios, torch.tensor(advantages_masked), ref_lp, policy_lp, clip_eps, kl_coef)

    adv_t = torch.tensor(advantages_masked)
    surr1 = ratios * adv_t
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    policy_obj = -torch.min(surr1, surr2).mean()
    kl = kl_penalty_term(ref_lp, policy_lp)
    expected = policy_obj + kl_coef * kl

    assert torch.allclose(loss, expected)
    masked_idx = next(i for i, m in enumerate(mask) if m)
    assert advantages_masked[masked_idx] == 0.0


def test_kl_penalty_term_shared_helper_identical():
    ref_lp = torch.tensor([math.log(0.5), math.log(0.25), math.log(0.75)])
    policy_lp = torch.tensor([math.log(0.4), math.log(0.5), math.log(0.1)])
    a = kl_penalty_term(ref_lp, policy_lp)
    b = kl_penalty_term(ref_lp, policy_lp)
    assert torch.allclose(a, b)
    assert a.dim() == 0
