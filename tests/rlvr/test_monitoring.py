"""Tests for the monitoring layer (Section 12)."""

import pytest

from rlvr.monitoring import (
    MonitoringDashboard,
    StepMetrics,
    compute_step_metrics,
    evaluate_validation_set,
    stage1_tool_use_filter,
    two_stage_tool_use_antihack,
)


def _metrics(step, **kwargs):
    return StepMetrics(step=step, **kwargs)


def test_dashboard_records_and_retrieves():
    dashboard = MonitoringDashboard()
    assert dashboard.latest() is None

    dashboard.record(_metrics(0, mean_reward=0.5))
    dashboard.record(_metrics(1, mean_reward=0.6))

    latest = dashboard.latest()
    assert latest is not None
    assert latest.step == 1
    assert len(dashboard.history) == 2


def test_dashboard_trend_extraction():
    dashboard = MonitoringDashboard()
    for s in range(5):
        dashboard.record(_metrics(s, mean_batch_entropy=1.0 - s * 0.2))

    trend = dashboard.trend("mean_batch_entropy")
    assert len(trend) == 5
    for actual, expected in zip(trend, [1.0, 0.8, 0.6, 0.4, 0.2]):
        assert actual == pytest.approx(expected, abs=1e-9)


def test_entropy_collapsing_detection():
    dashboard = MonitoringDashboard()
    for s in range(6):
        dashboard.record(_metrics(s, mean_batch_entropy=1.0 - s * 0.25))

    assert dashboard.entropy_collapsing(window=5, drop_threshold=0.3) is True


def test_entropy_not_collapsing_when_stable():
    dashboard = MonitoringDashboard()
    for s in range(6):
        dashboard.record(_metrics(s, mean_batch_entropy=1.0))

    assert dashboard.entropy_collapsing() is False


def test_entropy_collapsing_insufficient_history():
    dashboard = MonitoringDashboard()
    dashboard.record(_metrics(0, mean_batch_entropy=1.0))
    assert dashboard.entropy_collapsing() is False


def test_compute_step_metrics_basic():
    completions = [["hello world", "foo bar baz"]]
    rewards = [[0.5, 0.9]]
    metrics = compute_step_metrics(0, completions, rewards, batch_entropy=0.7, entropy_guard_trigger_pct=5.0)

    assert metrics.step == 0
    assert metrics.mean_reward == pytest.approx(0.7, abs=1e-6)
    assert metrics.reward_variance > 0.0
    assert metrics.zero_variance_group_pct == 0.0
    assert metrics.avg_response_length > 0.0
    assert metrics.mean_batch_entropy == 0.7
    assert metrics.entropy_guard_trigger_pct == 5.0


def test_compute_step_metrics_zero_variance_group():
    completions = [["a", "a"]]
    rewards = [[0.0, 0.0]]
    metrics = compute_step_metrics(0, completions, rewards)
    assert metrics.zero_variance_group_pct == 100.0


def test_compute_step_metrics_empty():
    metrics = compute_step_metrics(0, [], [])
    assert metrics.mean_reward == 0.0
    assert metrics.zero_variance_group_pct == 0.0


def test_stage1_tool_use_filter_flags_hacks():
    assert stage1_tool_use_filter("cat eval/secret.txt") is True
    assert stage1_tool_use_filter("curl http://evil.com/solution") is True
    assert stage1_tool_use_filter("wget http://example.com/answer") is True


def test_stage1_tool_use_filter_allows_clean():
    assert stage1_tool_use_filter("read data.csv") is False
    assert stage1_tool_use_filter("compute sum of column") is False


def test_two_stage_tool_use_antihack_blocks_flagged():
    blocked, result = two_stage_tool_use_antihack("cat eval/secret.txt")
    assert blocked is True
    assert "BLOCKED" in result


def test_two_stage_tool_use_antihack_allows_clean():
    blocked, result = two_stage_tool_use_antihack("read data.csv")
    assert blocked is False
    assert result == "read data.csv"


def test_two_stage_tool_use_antihack_llm_judge_overrides():
    def judge(text):
        return False
    blocked, result = two_stage_tool_use_antihack("cat eval/secret.txt", llm_judge=judge)
    assert blocked is False
    assert result == "cat eval/secret.txt"


def test_evaluate_validation_set():
    def reward_fn(c, gt, dom):
        return 1.0 if c == gt else 0.0

    completions = ["a", "b", "c"]
    ground_truths = ["a", "x", "c"]
    domains = ["math", "math", "math"]
    score = evaluate_validation_set(completions, ground_truths, domains, reward_fn)
    assert score == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_entropy_exactly_zero_detection():
    dashboard = MonitoringDashboard()
    dashboard.record(_metrics(0, mean_batch_entropy=1.0))
    dashboard.record(_metrics(1, mean_batch_entropy=0.0))
    assert dashboard.entropy_exactly_zero(consecutive=2) is False
    dashboard.record(_metrics(2, mean_batch_entropy=0.0))
    assert dashboard.entropy_exactly_zero(consecutive=2) is True


def test_format_reward_collapsed_detection():
    dashboard = MonitoringDashboard()
    dashboard.record(_metrics(0, format_reward_mean=0.2))
    dashboard.record(_metrics(1, format_reward_mean=0.0))
    assert dashboard.format_reward_collapsed(consecutive=2) is False
    dashboard.record(_metrics(2, format_reward_mean=0.0))
    assert dashboard.format_reward_collapsed(consecutive=2) is True


def test_record_live_signals_format_and_entropy_alerts():
    dashboard = MonitoringDashboard()
    a1 = dashboard.record_live_signals(
        entropy=0.0, format_mean=0.0, format_std=0.0, step=1, consecutive=2
    )
    assert a1["format_collapsed"] is False
    assert a1["entropy_zero"] is False

    a2 = dashboard.record_live_signals(
        entropy=0.0, format_mean=0.0, format_std=0.0, step=2, consecutive=2
    )
    assert a2["format_collapsed"] is True
    assert a2["entropy_zero"] is True
    assert dashboard.format_collapse_alerts == 1
    assert dashboard.entropy_zero_alerts == 1


def test_record_live_signals_resets_streaks_on_recovery():
    dashboard = MonitoringDashboard()
    dashboard.record_live_signals(entropy=0.0, format_mean=0.0, format_std=0.0, step=1)
    dashboard.record_live_signals(entropy=0.5, format_mean=0.3, format_std=0.1, step=2)
    assert dashboard.entropy_zero_streak == 0
    assert dashboard.format_zero_streak == 0


def test_evaluate_validation_set_empty():
    assert evaluate_validation_set([], [], [], lambda *a: 1.0) == 0.0
