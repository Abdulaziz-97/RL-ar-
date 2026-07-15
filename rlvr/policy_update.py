"""
Module 8 — Policy Update (GRPO/GSPO-switchable).

Clipped surrogate objective with shared KL penalty wrapper.
Consumes entropy-guarded loss terms from Module 7.
"""

import torch


def kl_penalty_term(ref_log_probs, policy_log_probs):
    """Schulman k3 KL estimator for on-policy samples drawn from the current policy.

    Computes E[exp(r) * r - (exp(r) - 1)] where r = log π_ref - log π_policy,
    which is an unbiased, low-variance approximation of KL(π_policy || π_ref).
    """
    log_ratio = ref_log_probs - policy_log_probs
    ratio = torch.exp(log_ratio)
    return (ratio * log_ratio - (ratio - 1)).mean()


def grpo_loss(ratios, advantages, ref_log_probs, policy_log_probs, clip_eps: float = 0.2, kl_coef: float = 0.04):
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = -torch.min(surr1, surr2).mean()
    kl = kl_penalty_term(ref_log_probs, policy_log_probs)
    return policy_obj + kl_coef * kl


def gspo_loss(sequence_ratios, advantages, ref_log_probs, policy_log_probs, clip_eps: float = 0.2, kl_coef: float = 0.04):
    surr1 = sequence_ratios * advantages
    surr2 = torch.clamp(sequence_ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = -torch.min(surr1, surr2).mean()
    kl = kl_penalty_term(ref_log_probs, policy_log_probs)
    return policy_obj + kl_coef * kl
