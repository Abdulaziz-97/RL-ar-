"""Tests for the curriculum sampler and callback."""

import pytest

from rlvr_pipeline.curriculum_sampler import CurriculumSampler, CurriculumCallback


def test_curriculum_sampler_initializes():
    tags = ["trivial", "easy", "medium", "hard"] * 2
    sampler = CurriculumSampler(tags, total_steps=100, schedule_type="gaussian", sigma_fraction=0.2)
    assert len(sampler) == 8


def test_curriculum_sampler_len_matches_tags():
    tags = ["trivial"] * 5 + ["easy"] * 5 + ["medium"] * 5 + ["hard"] * 5
    sampler = CurriculumSampler(tags, total_steps=60, schedule_type="gaussian", sigma_fraction=0.2)
    assert len(sampler) == 20


def test_curriculum_sampler_set_step():
    tags = ["trivial"] * 4 + ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4
    sampler = CurriculumSampler(tags, total_steps=40, schedule_type="gaussian", sigma_fraction=0.2)
    sampler.set_step(10)
    # should not raise; step is stored internally
    sampler.set_step(30)


def test_curriculum_sampler_bucket_distribution_early():
    tags = ["trivial"] * 10 + ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
    sampler = CurriculumSampler(tags, total_steps=120, schedule_type="gaussian", sigma_fraction=0.2)
    sampler.set_step(0)
    dist = sampler.get_bucket_distribution()
    # early step: dominated by trivial/easy
    assert dist.get("trivial", 0) + dist.get("easy", 0) > 0.9


def test_curriculum_sampler_bucket_distribution_mid():
    tags = ["trivial"] * 10 + ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
    sampler = CurriculumSampler(tags, total_steps=120, schedule_type="gaussian", sigma_fraction=0.2)
    sampler.set_step(60)
    dist = sampler.get_bucket_distribution()
    # mid step: even-ish across all buckets
    for b in ["trivial", "easy", "medium", "hard"]:
        assert b in dist


def test_curriculum_sampler_bucket_distribution_late():
    tags = ["trivial"] * 10 + ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
    sampler = CurriculumSampler(tags, total_steps=120, schedule_type="gaussian", sigma_fraction=0.2)
    sampler.set_step(120)
    dist = sampler.get_bucket_distribution()
    # late step: dominated by mediums/hard
    assert dist.get("medium", 0) + dist.get("hard", 0) > 0.5


def test_curriculum_sampler_fixed_switch():
    tags = ["trivial"] * 4 + ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4
    sampler = CurriculumSampler(tags, total_steps=32, schedule_type="fixed_switch", sigma_fraction=0.25)
    sampler.set_step(0)
    early_dist = sampler.get_bucket_distribution()
    sampler.set_step(16)
    mid_dist = sampler.get_bucket_distribution()
    # fixed_switch should change distribution at step=total_steps*0.25 and step=total_steps*0.75
    assert early_dist != mid_dist


def test_curriculum_sampler_random_mix():
    tags = ["trivial"] * 4 + ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4
    sampler = CurriculumSampler(tags, total_steps=16, schedule_type="random_mix", sigma_fraction=0.2)
    sampler.set_step(5)
    dist = sampler.get_bucket_distribution()
    # random mix: roughly uniform distribution across all buckets
    total = sum(dist.values())
    for b in ["trivial", "easy", "medium", "hard"]:
        assert b in dist


def test_curriculum_sampler_none_schedule():
    tags = ["trivial"] * 4 + ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4
    sampler = CurriculumSampler(tags, total_steps=16, schedule_type="none", sigma_fraction=0.2)
    sampler.set_step(5)
    dist = sampler.get_bucket_distribution()
    assert isinstance(dist, dict)
    assert len(dist) == 0, f"'none' schedule returns empty dict, got: {dist}"


def test_curriculum_sampler_iterates():
    tags = ["easy"] * 4 + ["hard"] * 4
    sampler = CurriculumSampler(tags, total_steps=16, schedule_type="gaussian", sigma_fraction=0.3)
    sampler.set_step(0)
    indices = list(iter(sampler))
    assert len(indices) == 8


def test_curriculum_callback():
    tags = ["easy"] * 4
    sampler = CurriculumSampler(tags, total_steps=16, schedule_type="gaussian", sigma_fraction=0.3)
    callback = CurriculumCallback(sampler)

    mock_args = type("MockArgs", (), {"logging_steps": 1})()
    mock_state = type("MockState", (), {"global_step": 0})()
    mock_control = type("MockControl", (), {})()
    mock_trainer = type("MockTrainer", (), {})()
    callback.on_train_begin(mock_args, mock_state, mock_control, trainer=mock_trainer)
    assert sampler._current_step == 0

    mock_state.global_step = 5
    callback.on_step_begin(mock_args, mock_state, mock_control, trainer=mock_trainer)
    assert sampler._current_step == 5
