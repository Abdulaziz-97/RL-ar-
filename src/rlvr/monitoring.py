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
    format_reward_mean: float = 0.0
    validation_reward: Optional[float] = None


@dataclass
class MonitoringDashboard:
    history: list[StepMetrics] = field(default_factory=list)
    # Live-run alert counters (updated via record_live_signals / helpers below)
    format_zero_streak: int = 0
    entropy_zero_streak: int = 0
    format_collapse_alerts: int = 0
    entropy_zero_alerts: int = 0

    def record(self, metrics: StepMetrics) -> None:
        self.history.append(metrics)

    def latest(self) -> Optional[StepMetrics]:
        return self.history[-1] if self.history else None

    def trend(self, metric_name: str) -> list[float]:
        return [getattr(m, metric_name) for m in self.history if getattr(m, metric_name) is not None]

    def entropy_collapsing(self, window: int = 5, drop_threshold: float = 0.3) -> bool:
        """True when entropy drops sharply *and* is entering a low regime.

        Absolute drops of 0.3 nats from healthy levels (e.g. 6.7→6.3) are
        normal GRPO noise and must not alert. Require both a large relative
        drop and a low absolute floor.
        """
        entropies = self.trend("mean_batch_entropy")
        if len(entropies) < 2:
            return False
        recent = entropies[-window:]
        if len(recent) < 2:
            return False
        start, end = recent[0], recent[-1]
        if start <= 0:
            return end == 0.0
        relative_drop = (start - end) / start
        # Alert only if entropy fell by ≥40% of its window start AND is now < 2.0
        return relative_drop >= 0.4 and end < 2.0

    def entropy_exactly_zero(self, consecutive: int = 2) -> bool:
        """True when the last `consecutive` logged entropies are exactly 0."""
        entropies = self.trend("mean_batch_entropy")
        if len(entropies) < consecutive:
            return False
        return all(e == 0.0 for e in entropies[-consecutive:])

    def format_reward_collapsed(self, consecutive: int = 2) -> bool:
        """True when format mean has been exactly 0 for `consecutive` logs.

        Leading indicator of systemic truncation (clipped before </answer>),
        not natural per-batch variance.
        """
        formats = self.trend("format_reward_mean")
        if len(formats) < consecutive:
            return False
        return all(f == 0.0 for f in formats[-consecutive:])

    def record_live_signals(
        self,
        *,
        entropy: Optional[float] = None,
        format_mean: Optional[float] = None,
        format_std: Optional[float] = None,
        step: Optional[int] = None,
        consecutive: int = 2,
    ) -> dict[str, bool]:
        """Update streaks from live TRL log dicts; return alert flags.

        Call this from a TrainerCallback on every logging step.
        """
        alerts = {
            "format_collapsed": False,
            "entropy_zero": False,
            "entropy_collapsing": False,
        }

        if format_mean is not None:
            # Collapse = every completion failed format (mean 0, std ~0).
            is_collapsed = format_mean == 0.0 and (
                format_std is None or format_std == 0.0
            )
            self.format_zero_streak = self.format_zero_streak + 1 if is_collapsed else 0
            if self.format_zero_streak >= consecutive:
                alerts["format_collapsed"] = True
                self.format_collapse_alerts += 1

        if entropy is not None:
            self.entropy_zero_streak = (
                self.entropy_zero_streak + 1 if entropy == 0.0 else 0
            )
            if self.entropy_zero_streak >= consecutive:
                alerts["entropy_zero"] = True
                self.entropy_zero_alerts += 1

        # Keep StepMetrics history so trend helpers work for live runs.
        if step is not None and (entropy is not None or format_mean is not None):
            self.record(
                StepMetrics(
                    step=step,
                    mean_batch_entropy=entropy if entropy is not None else 0.0,
                    format_reward_mean=format_mean if format_mean is not None else 0.0,
                )
            )
            if entropy is not None:
                alerts["entropy_collapsing"] = self.entropy_collapsing()

        return alerts


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
