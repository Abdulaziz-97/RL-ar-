"""
Integration test (Section 11) — end-to-end mock pipeline through all 8 modules.

Validates orchestration correctness, not model quality. Uses a stub model.
Runs in CI on every commit.
"""

import math

import pytest
import torch

from rlvr.config import PipelineConfig
from rlvr.failure_bank import clear_bank
from rlvr.monitoring import MonitoringDashboard
from rlvr.pipeline import run_training_step


class StubModel:
    def __init__(self, seed: int = 42):
        self._rng_seed = seed

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        if temperature == 0.0:
            return "<think>\u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629</think><answer>4</answer>"
        h = hash(f"{self._rng_seed}:{prompt}:{temperature}") % 10
        if h < 7:
            return "<think>\u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629</think><answer>4</answer>"
        return "<think>\u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u062b\u0645 \u0623\u062d\u0633\u0628</think><answer>5</answer>"


def _make_samples(n: int = 4):
    return [
        {
            "id": f"math-{i}",
            "domain": "math",
            "difficulty_tag": "easy",
            "prompt": f"What is 2+2? variant {i}",
            "ground_truth_answer": 4,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _reset_bank():
    clear_bank()
    yield
    clear_bank()


def test_full_mock_pipeline_completes_without_crash():
    config = PipelineConfig(group_size=4)
    model = StubModel()
    samples = _make_samples(4)

    result = run_training_step(config, model, samples, step=0)

    assert isinstance(result.loss, float)
    assert not math.isnan(result.loss)
    assert not math.isinf(result.loss)
    assert len(result.advantages) == 4 * 4
    assert not result.has_nan
    assert all(not (math.isnan(a) or math.isinf(a)) for a in result.advantages)


def test_pipeline_produces_advantage_tensor_no_nans():
    config = PipelineConfig(group_size=8)
    model = StubModel()
    samples = _make_samples(3)

    result = run_training_step(config, model, samples, step=1)

    assert len(result.advantages) == 3 * 8
    assert not any(math.isnan(a) or math.isinf(a) for a in result.advantages)


def test_config_switches_grpo_to_gspo_without_code_change():
    model = StubModel()
    samples = _make_samples(4)

    grpo_config = PipelineConfig(group_size=4, policy_update_mode="grpo")
    gspo_config = PipelineConfig(group_size=4, policy_update_mode="gspo")

    grpo_result = run_training_step(grpo_config, model, samples, step=0)
    clear_bank()
    gspo_result = run_training_step(gspo_config, model, samples, step=0)

    assert not grpo_result.has_nan
    assert not gspo_result.has_nan
    assert isinstance(grpo_result.loss, float)
    assert isinstance(gspo_result.loss, float)


def test_config_switches_clip_cov_to_kl_cov_without_code_change():
    model = StubModel()
    samples = _make_samples(4)

    clip_config = PipelineConfig(group_size=4, entropy_guard_mode="clip_cov")
    kl_config = PipelineConfig(group_size=4, entropy_guard_mode="kl_cov")

    clip_result = run_training_step(clip_config, model, samples, step=0)
    clear_bank()
    kl_result = run_training_step(kl_config, model, samples, step=0)

    assert not clip_result.has_nan
    assert not kl_result.has_nan
    assert clip_result.metrics.entropy_guard_trigger_pct >= 0.0
    assert kl_result.metrics.entropy_guard_trigger_pct >= 0.0


def test_config_switches_easy_to_hard_curriculum_without_code_change():
    model = StubModel()
    samples = _make_samples(4)

    easy_config = PipelineConfig(group_size=4, curriculum_stage="easy")
    hard_config = PipelineConfig(group_size=4, curriculum_stage="hard")

    easy_result = run_training_step(easy_config, model, samples, step=0)
    hard_result = run_training_step(hard_config, model, samples, step=0)

    assert not easy_result.has_nan
    assert not hard_result.has_nan


def test_monitoring_dashboard_records_metrics():
    dashboard = MonitoringDashboard()
    model = StubModel()
    samples = _make_samples(4)

    for step in range(3):
        config = PipelineConfig(group_size=4)
        result = run_training_step(config, model, samples, step=step)
        dashboard.record(result.metrics)

    assert len(dashboard.history) == 3
    latest = dashboard.latest()
    assert latest is not None
    assert latest.step == 2
    assert len(dashboard.trend("mean_reward")) == 3
    assert all(0.0 <= r <= 1.0 for r in dashboard.trend("mean_reward"))


def test_pipeline_with_validation_set():
    model = StubModel()
    samples = _make_samples(2)
    validation_set = {
        "prompts": ["What is 2+2?"],
        "ground_truths": [4],
        "domains": ["math"],
    }

    config = PipelineConfig(group_size=4)
    result = run_training_step(config, model, samples, step=0, validation_set=validation_set)

    assert result.metrics.validation_reward is not None
    assert 0.0 <= result.metrics.validation_reward <= 1.0


def test_pipeline_empty_samples_does_not_crash():
    config = PipelineConfig(group_size=4)
    model = StubModel()
    result = run_training_step(config, model, [], step=0)
    assert not result.has_nan
    assert result.loss == 0.0


def test_all_zero_reward_group_gets_banked():
    class AlwaysWrongModel:
        def generate(self, prompt, temperature, max_new_tokens):
            return "<think></think><answer>999</answer>"

    config = PipelineConfig(group_size=4)
    model = AlwaysWrongModel()
    samples = _make_samples(2)

    result = run_training_step(config, model, samples, step=0)
    assert result.banked_count == 2
