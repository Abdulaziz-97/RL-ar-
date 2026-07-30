"""Tests for the CRPS (Curriculum Replay via Progressive Suffixes) module."""

import pytest

from rlvr.crps import (
    SuccessTrace,
    SuccessTraceStore,
    build_progressive_suffix,
    find_related_success_trace,
    get_crps_suffix_fraction,
    should_activate_crps,
)


def _make_trace(
    prompt_id: str = "p1",
    puzzle_type: str = "arithmetic",
    difficulty_tag: str = "hard",
    token_sequence: list[str] | None = None,
    step_recorded: int = 0,
) -> SuccessTrace:
    return SuccessTrace(
        prompt_id=prompt_id,
        puzzle_type=puzzle_type,
        difficulty_tag=difficulty_tag,
        token_sequence=token_sequence if token_sequence is not None else ["x"],
        step_recorded=step_recorded,
    )


def test_find_related_success_trace_respects_puzzle_type_and_difficulty():
    traces = [
        _make_trace(prompt_id="a", puzzle_type="arithmetic", difficulty_tag="hard", step_recorded=0),
        _make_trace(prompt_id="b", puzzle_type="arithmetic", difficulty_tag="medium", step_recorded=1),
        _make_trace(prompt_id="c", puzzle_type="logic", difficulty_tag="hard", step_recorded=2),
        _make_trace(prompt_id="d", puzzle_type="logic", difficulty_tag="medium", step_recorded=3),
    ]

    result = find_related_success_trace(
        traces, puzzle_type="arithmetic", difficulty_tag="hard",
        max_age_steps=100, current_step=50,
    )
    assert result is not None
    assert result.prompt_id == "a"

    result = find_related_success_trace(
        traces, puzzle_type="logic", difficulty_tag="medium",
        max_age_steps=100, current_step=50,
    )
    assert result is not None
    assert result.prompt_id == "d"

    result = find_related_success_trace(
        traces, puzzle_type="arithmetic", difficulty_tag="medium",
        max_age_steps=100, current_step=50,
    )
    assert result is not None
    assert result.prompt_id == "b"


def test_find_related_success_trace_respects_max_age_steps():
    traces = [_make_trace(prompt_id="old", step_recorded=10)]

    # age = 20 - 10 = 10 > max_age_steps=5 -> not returned
    result = find_related_success_trace(
        traces, puzzle_type="arithmetic", difficulty_tag="hard",
        max_age_steps=5, current_step=20,
    )
    assert result is None

    # age = 14 - 10 = 4 <= max_age_steps=5 -> returned
    result = find_related_success_trace(
        traces, puzzle_type="arithmetic", difficulty_tag="hard",
        max_age_steps=5, current_step=14,
    )
    assert result is not None
    assert result.prompt_id == "old"


def test_find_related_success_trace_returns_none_when_no_match():
    traces = [
        _make_trace(prompt_id="a", puzzle_type="arithmetic", difficulty_tag="hard", step_recorded=0),
    ]

    result = find_related_success_trace(
        traces, puzzle_type="nonexistent", difficulty_tag="hard",
        max_age_steps=100, current_step=10,
    )
    assert result is None

    result = find_related_success_trace(
        [], puzzle_type="arithmetic", difficulty_tag="hard",
        max_age_steps=100, current_step=10,
    )
    assert result is None


def test_find_related_success_trace_returns_most_recent_match():
    traces = [
        _make_trace(prompt_id="first", puzzle_type="arithmetic", difficulty_tag="hard", step_recorded=0),
        _make_trace(prompt_id="second", puzzle_type="arithmetic", difficulty_tag="hard", step_recorded=5),
        _make_trace(prompt_id="third", puzzle_type="arithmetic", difficulty_tag="hard", step_recorded=10),
    ]

    result = find_related_success_trace(
        traces, puzzle_type="arithmetic", difficulty_tag="hard",
        max_age_steps=100, current_step=50,
    )
    assert result is not None
    assert result.prompt_id == "third"


def test_progressive_suffix_length_scales_with_fraction():
    tokens = [f"tok_{i}" for i in range(100)]
    trace = _make_trace(token_sequence=tokens)

    assert build_progressive_suffix(trace, 0.1) == " ".join(tokens[-10:])
    assert build_progressive_suffix(trace, 0.3) == " ".join(tokens[-30:])
    assert build_progressive_suffix(trace, 0.5) == " ".join(tokens[-50:])
    assert build_progressive_suffix(trace, 1.0) == " ".join(tokens[-100:])


def test_progressive_suffix_minimum_one_token():
    tokens = [f"tok_{i}" for i in range(100)]
    trace = _make_trace(token_sequence=tokens)

    result = build_progressive_suffix(trace, 0.0)
    assert result == tokens[-1]


def test_crps_only_activates_in_hard_curriculum_stage():
    assert should_activate_crps("hard", True, 1) is True
    assert should_activate_crps("easy", True, 1) is False
    assert should_activate_crps("medium", True, 1) is False
    assert should_activate_crps("trivial", True, 1) is False


def test_crps_requires_at_least_one_prior_success_or_returns_none():
    assert should_activate_crps("hard", False, 1) is False
    assert should_activate_crps("hard", True, 0) is False


def test_get_crps_suffix_fraction_schedule():
    assert get_crps_suffix_fraction(0) == 0.0
    assert get_crps_suffix_fraction(1) == 0.1
    assert get_crps_suffix_fraction(2) == 0.3
    assert get_crps_suffix_fraction(3) == 0.5
    assert get_crps_suffix_fraction(4) == 0.5
    assert get_crps_suffix_fraction(10) == 0.5


def test_success_trace_store_eviction_is_fifo():
    store = SuccessTraceStore(max_traces=3)
    for i in range(5):
        store.add(_make_trace(prompt_id=f"p{i}"))

    assert len(store) == 3
    remaining_ids = [t.prompt_id for t in store.traces]
    assert remaining_ids == ["p2", "p3", "p4"]


def test_success_trace_store_clear():
    store = SuccessTraceStore(max_traces=10)
    store.add(_make_trace(prompt_id="p1"))
    store.add(_make_trace(prompt_id="p2"))
    assert len(store) == 2

    store.clear()

    assert len(store) == 0


def test_build_progressive_suffix_returns_correct_tokens():
    trace = _make_trace(token_sequence=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])

    assert build_progressive_suffix(trace, 0.3) == "h i j"
