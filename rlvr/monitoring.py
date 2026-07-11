"""
Module 12 — Anti-Hacking Monitoring Layer.

Tracks per-step metrics for the live dashboard described in Section 12:
  - mean reward
  - reward variance per group
  - percentage of zero-variance groups
  - average response length (length-hacking early warning)
  - Arabic word-ratio trend
  - mean batch entropy (entropy-collapse early warning)
  - percentage of tokens excluded by Clip-Cov / mean KL-Cov penalty
  - answer-leak and structural-leak penalty trigger rates

Also provides a held-out validation-set checker and a two-stage tool-use
anti-hack filter for logic samples.
"""

from dataclasses import dataclass, field
from typing import Optional

from rlvr.reward_composer import (
    DEFAULT_LEAK_PHRASES,
    penalty_answer_leak,
    penalty_structural_leak,
    reward_language_consistency,
)


@dataclass
class StepMetrics:
    step: int
    mean_reward: float = 0.0
    reward_variance: float = 0.0
    zero_variance_group_pct: float = 0.0
    avg_response_length: float = 0.0
    arabic_word_ratio: float = 0.0
    mean_batch_entropy: float = 0.0
    entropy_guard_trigger_pct: float = 0.0
    answer_leak_trigger_rate: float = 0.0
    structural_leak_trigger_rate: float = 0.0
    validation_reward: Optional[float] = None


@dataclass
class MonitoringDashboard:
    history: list[StepMetrics] = field(default_factory=list)

    def record(self, metrics: StepMetrics) -> None:
        self.history.append(metrics)

    def latest(self) -> Optional[StepMetrics]:
        return self.history[-1] if self.history else None

    def trend(self, metric_name: str) -> list[float]:
        return [getattr(m, metric_name) for m in self.history if getattr(m, metric_name) is not None]

    def entropy_collapsing(self, window: int = 5, drop_threshold: float = 0.3) -> bool:
        entropies = self.trend("mean_batch_entropy")
        if len(entropies) < 2:
            return False
        recent = entropies[-window:]
        if len(recent) < 2:
            return False
        return (recent[0] - recent[-1]) > drop_threshold


def compute_step_metrics(
    step: int,
    completions: list[list[str]],
    rewards: list[list[float]],
    batch_entropy: float = 0.0,
    entropy_guard_trigger_pct: float = 0.0,
    validation_reward: Optional[float] = None,
) -> StepMetrics:
    flat_rewards = [r for group in rewards for r in group]
    n = len(flat_rewards) if flat_rewards else 1
    mean_reward = sum(flat_rewards) / n if flat_rewards else 0.0
    reward_variance = (
        sum((r - mean_reward) ** 2 for r in flat_rewards) / n if flat_rewards else 0.0
    )

    zero_var_groups = sum(1 for group in rewards if len(set(group)) <= 1)
    zero_variance_group_pct = (
        (zero_var_groups / len(rewards) * 100) if rewards else 0.0
    )

    flat_completions = [c for group in completions for c in group]
    avg_response_length = (
        sum(len(c.split()) for c in flat_completions) / len(flat_completions)
        if flat_completions
        else 0.0
    )
    arabic_word_ratio = (
        sum(reward_language_consistency(c) for c in flat_completions) / len(flat_completions)
        if flat_completions
        else 0.0
    )

    leak_triggers = sum(
        penalty_answer_leak(c, None, DEFAULT_LEAK_PHRASES) for c in flat_completions
    )
    struct_triggers = sum(penalty_structural_leak(c) for c in flat_completions)
    total = len(flat_completions) if flat_completions else 1
    answer_leak_trigger_rate = leak_triggers / total
    structural_leak_trigger_rate = struct_triggers / total

    return StepMetrics(
        step=step,
        mean_reward=mean_reward,
        reward_variance=reward_variance,
        zero_variance_group_pct=zero_variance_group_pct,
        avg_response_length=avg_response_length,
        arabic_word_ratio=arabic_word_ratio,
        mean_batch_entropy=batch_entropy,
        entropy_guard_trigger_pct=entropy_guard_trigger_pct,
        answer_leak_trigger_rate=answer_leak_trigger_rate,
        structural_leak_trigger_rate=structural_leak_trigger_rate,
        validation_reward=validation_reward,
    )


_TOOL_HACK_PATTERNS = ("eval", "secret", "hidden", "answer", "curl", "wget", "solution")
_EXTERNAL_CALL_PATTERNS = ("curl ", "wget ", "http://", "https://")


def stage1_tool_use_filter(action_text: str) -> bool:
    lowered = action_text.lower()
    for pat in _TOOL_HACK_PATTERNS:
        if pat in lowered:
            return True
    for pat in _EXTERNAL_CALL_PATTERNS:
        if pat in lowered:
            return True
    return False


def two_stage_tool_use_antihack(action_text: str, llm_judge=None) -> tuple[bool, str]:
    flagged = stage1_tool_use_filter(action_text)
    if not flagged:
        return False, action_text
    if llm_judge is not None:
        is_hack = llm_judge(action_text)
        if not is_hack:
            return False, action_text
    return True, "[BLOCKED: tool-use anti-hack]"


def evaluate_validation_set(completions, ground_truths, domains, reward_fn) -> float:
    if not completions:
        return 0.0
    total = sum(
        reward_fn(c, gt, dom) for c, gt, dom in zip(completions, ground_truths, domains)
    )
    return total / len(completions)
