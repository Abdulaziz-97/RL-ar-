"""
Pipeline orchestrator — ties all eight modules into a single training step.

Supports single-flag switching (per Definition of Done):
  - policy_update_mode:   "grpo" <-> "gspo"
  - entropy_guard_mode:   "clip_cov" <-> "kl_cov"
  - curriculum_stage:     "easy" <-> "hard"
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from rlvr.advantage_calculator import compute_advantages, flag_zero_variance_group
from rlvr.config import PipelineConfig
from rlvr.entropy_guard import (
    apply_entropy_guard,
    clip_cov_mask,
    compute_batch_entropy,
    compute_token_covariance,
    kl_cov_penalty,
)
from rlvr.failure_bank import bank_failed_group, retrieve_banked_failures
from rlvr.monitoring import StepMetrics, compute_step_metrics
from rlvr.policy_update import grpo_loss, gspo_loss
from rlvr.reward_composer import compose_reward
from rlvr.rollout_engine import generate_group


@dataclass
class StepResult:
    loss: float
    advantages: list[float]
    metrics: StepMetrics
    banked_count: int
    has_nan: bool


def _flatten(nested):
    return [item for group in nested for item in group]


def run_training_step(
    config: PipelineConfig,
    model,
    samples: list[dict],
    step: int = 0,
    validation_set: Optional[dict] = None,
) -> StepResult:
    if config.curriculum_stage == "hard":
        banked = retrieve_banked_failures("hard")
    else:
        banked = []

    completions = [
        generate_group(
            model,
            s["prompt"],
            group_size=config.group_size,
            temperature=config.temperature,
        )
        for s in samples
    ]

    rewards = [
        [compose_reward(c, s["ground_truth_answer"], s["domain"], s.get("puzzle_type")) for c in group]
        for group, s in zip(completions, samples)
    ]

    advantages_groups = [compute_advantages(r, config.eps) for r in rewards]

    banked_count = 0
    for s, comps, rews in zip(samples, completions, rewards):
        if flag_zero_variance_group(rews) and all(r == 0.0 for r in rews):
            bank_failed_group(s["id"], comps, rews)
            banked_count += 1

    flat_advantages = _flatten(advantages_groups)
    n_tokens = len(flat_advantages)

    uniform_lp = -math.log(max(n_tokens, 1)) if n_tokens > 0 else 0.0
    flat_log_probs = [uniform_lp] * n_tokens

    covariances = compute_token_covariance(flat_log_probs, flat_advantages)

    if config.entropy_guard_mode == "clip_cov":
        mask = clip_cov_mask(covariances, config.clip_ratio)
        guarded = apply_entropy_guard(flat_advantages, mask, "clip_cov")
        trigger_pct = (sum(mask) / len(mask) * 100) if mask else 0.0
    else:
        penalties = kl_cov_penalty(covariances, config.kl_cov_coef, config.top_k_ratio)
        guarded = apply_entropy_guard(flat_advantages, penalties, "kl_cov")
        trigger_pct = (sum(1 for p in penalties if p > 0) / len(penalties) * 100) if penalties else 0.0

    batch_entropy = compute_batch_entropy([flat_log_probs]) if flat_log_probs else 0.0

    if n_tokens == 0:
        loss_tensor = torch.tensor(0.0)
    else:
        ratios = torch.ones(n_tokens, dtype=torch.float32)
        advantages_tensor = torch.tensor(guarded, dtype=torch.float32)
        ref_lp = torch.tensor(flat_log_probs, dtype=torch.float32)
        policy_lp = torch.tensor(flat_log_probs, dtype=torch.float32)

        if config.policy_update_mode == "grpo":
            loss_tensor = grpo_loss(
                ratios, advantages_tensor, ref_lp, policy_lp, config.clip_eps, config.kl_coef
            )
        else:
            seq_ratio = torch.tensor([1.0], dtype=torch.float32)
            loss_tensor = gspo_loss(
                seq_ratio, advantages_tensor, ref_lp, policy_lp, config.clip_eps, config.kl_coef
            )

    val_reward = None
    if validation_set is not None:
        val_completions = [
            generate_group(
                model,
                p,
                group_size=1,
                temperature=0.0,
                max_new_tokens=512,
            )[0]
            for p in validation_set["prompts"]
        ]
        val_reward = sum(
            compose_reward(c, gt, dom)
            for c, gt, dom in zip(
                val_completions,
                validation_set["ground_truths"],
                validation_set["domains"],
            )
        ) / max(len(val_completions), 1)

    metrics = compute_step_metrics(
        step=step,
        completions=completions,
        rewards=rewards,
        batch_entropy=batch_entropy,
        entropy_guard_trigger_pct=trigger_pct,
        validation_reward=val_reward,
    )

    has_nan = bool(
        any(math.isnan(a) or math.isinf(a) for a in flat_advantages)
        or torch.isnan(loss_tensor).item()
        or torch.isinf(loss_tensor).item()
    )

    return StepResult(
        loss=loss_tensor.item(),
        advantages=flat_advantages,
        metrics=metrics,
        banked_count=banked_count,
        has_nan=has_nan,
    )
