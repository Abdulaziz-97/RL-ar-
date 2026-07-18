"""
Module — Failure Mining (RL-ZVP + POPO).

Combines two failure-mining strategies for zero-variance / low-variance groups:

  * RL-ZVP direct scoring: when a group's reward variance is zero, GRPO's
    group-relative advantage collapses to zero and no gradient signal flows.
    This module produces a direct per-token loss-weight signal so that
    all-wrong groups still penalise confident-but-wrong tokens, and
    all-correct groups still reward low-confidence-but-correct tokens.

  * POPO prioritized replay: bank groups with non-zero reward variance
    (the "effective" groups that carry real GRPO signal) and replay them
    in place of future ineffective (zero-variance) groups. Uses decoupled
    importance sampling so the PPO clip only acts on the recent policy
    while the true origin-policy correction is applied as a multiplicative
    factor outside the clip.
"""

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass


def is_zero_variance_group(rewards: list[float], eps: float = 1e-6) -> bool:
    """True if every completion in the group received the same reward."""
    return (max(rewards) - min(rewards)) < eps


def score_zero_variance_group(
    completions: list[str],
    rewards: list[float],
    token_confidences: list[list[float]],
) -> list[list[float]]:
    """
    Returns per-token loss-weight signals for a zero-variance group.
    Only call this when is_zero_variance_group(rewards) is True.
    """
    all_wrong = rewards[0] == 0.0
    if all_wrong:
        return [[-conf for conf in confs] for confs in token_confidences]
    return [[1.0 - conf for conf in confs] for confs in token_confidences]


def is_effective_group(rewards: list[float], eps: float = 1e-6) -> bool:
    """Any group with non-zero reward variance counts as effective — including partially-correct groups, NOT just all-zero failures."""
    return statistics.pstdev(rewards) > eps


@dataclass
class ReplayEntry:
    prompt_id: str
    completions: list[str]
    rewards: list[float]
    old_policy_log_probs: list[list[float]]
    step_generated: int


class ReplayBuffer:
    def __init__(self, max_size: int = 512):
        self.buffer: deque[ReplayEntry] = deque(maxlen=max_size)

    def push(self, entry: ReplayEntry) -> None:
        self.buffer.append(entry)

    def sample(self) -> ReplayEntry | None:
        if len(self.buffer) == 0:
            return None
        return random.choice(list(self.buffer))

    def __len__(self) -> int:
        return len(self.buffer)


def maybe_push_to_buffer(buffer: ReplayBuffer, entry: ReplayEntry) -> None:
    if is_effective_group(entry.rewards):
        buffer.push(entry)


def replace_ineffective_group(
    buffer: ReplayBuffer,
    current_group_rewards: list[float],
) -> ReplayEntry | None:
    if is_effective_group(current_group_rewards):
        return None
    return buffer.sample()


def decoupled_importance_ratio(
    current_log_probs: list[float],
    recent_log_probs: list[float],
    old_log_probs: list[float],
) -> tuple[list[float], list[float]]:
    """Returns (clip_ratios, correction_ratios) per token.

    clip_ratios = exp(current - recent)  — only this passes through PPO clip
    correction_ratios = exp(recent - old)  — applied as multiplicative factor outside clip
    """
    clip_ratios = [math.exp(c - r) for c, r in zip(current_log_probs, recent_log_probs)]
    correction_ratios = [math.exp(r - o) for r, o in zip(recent_log_probs, old_log_probs)]
    return clip_ratios, correction_ratios
