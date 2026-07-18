"""Tests for Module 6 — Failure Bank."""

import pytest

from rlvr.failure_bank import (
    MAX_BANK_SIZE,
    bank_failed_group,
    bank_size,
    clear_bank,
    evict_stale_banked_samples,
    retrieve_banked_failures,
    set_current_stage,
)


@pytest.fixture(autouse=True)
def _reset_bank():
    clear_bank()
    set_current_stage("easy")
    yield
    clear_bank()
    set_current_stage("easy")


def test_only_all_zero_groups_get_banked_not_mixed_groups():
    before = bank_size()
    bank_failed_group("pA", ["c1", "c2"], [0.0, 0.0, 0.0])
    assert bank_size() == before + 1

    before = bank_size()
    bank_failed_group("pB", ["c1", "c2"], [0.0, 0.0, 0.5])
    assert bank_size() == before


def test_banked_samples_are_retrievable_in_hard_stage():
    set_current_stage("easy")
    bank_failed_group("p1", ["c1", "c2"], [0.0, 0.0, 0.0])

    hard_results = retrieve_banked_failures("hard")
    assert len(hard_results) == 1
    sample = hard_results[0]
    assert sample.prompt_id == "p1"
    assert sample.completions == ["c1", "c2"]
    assert sample.rewards == [0.0, 0.0, 0.0]
    assert sample.stage == "easy"

    assert retrieve_banked_failures("easy") == []


def test_bank_does_not_grow_unbounded():
    for i in range(MAX_BANK_SIZE + 50):
        bank_failed_group(f"p{i}", [f"c{i}"], [0.0, 0.0])

    assert bank_size() == MAX_BANK_SIZE


def test_evict_stale_banked_samples_removes_retired_prompts():
    bank_failed_group("p1", ["c"], [0.0, 0.0])
    bank_failed_group("p2", ["c"], [0.0, 0.0])
    bank_failed_group("p3", ["c"], [0.0, 0.0])

    removed = evict_stale_banked_samples({"p1", "p2"})
    assert removed == 1

    retrieved = retrieve_banked_failures("hard")
    prompt_ids = {s.prompt_id for s in retrieved}
    assert "p3" not in prompt_ids
    assert prompt_ids == {"p1", "p2"}


def test_retrieve_hard_stage_returns_all_banked():
    bank_failed_group("p1", ["c"], [0.0, 0.0])
    bank_failed_group("p2", ["c"], [0.0, 0.0])
    bank_failed_group("p3", ["c"], [0.0, 0.0])

    assert len(retrieve_banked_failures("hard")) == 3


def test_clear_bank_resets():
    bank_failed_group("p1", ["c"], [0.0, 0.0])
    bank_failed_group("p2", ["c"], [0.0, 0.0])
    assert bank_size() > 0

    clear_bank()

    assert bank_size() == 0


def test_non_zero_reward_not_banked():
    before = bank_size()
    bank_failed_group("pX", ["c1"], [0.0, 0.001, 0.0])
    assert bank_size() == before
