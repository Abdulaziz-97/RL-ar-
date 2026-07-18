"""Tests for Module 2 — Rollout Engine (Generation)."""

import pytest

from rlvr.rollout_engine import GenerativeModel, generate_group


class StubModel:
    def __init__(self, responses=None, deterministic_temp_zero=True):
        self.calls = 0
        self.responses = responses or []
        self._det = deterministic_temp_zero

    def generate(self, prompt, temperature, max_new_tokens):
        self.calls += 1
        if temperature == 0.0 and self._det:
            return "deterministic-response"
        if self.responses:
            return self.responses[self.calls % len(self.responses)]
        return f"response-{self.calls}"


class RaisingOnEmptyModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, temperature, max_new_tokens):
        self.calls += 1
        if prompt == "":
            raise ValueError("empty prompt")
        return f"response-{self.calls}"


def test_returns_exactly_group_size_completions():
    model = StubModel()
    result = generate_group(model, "مرحبا", group_size=8)
    assert len(result) == 8
    assert all(isinstance(r, str) for r in result)


def test_completions_respect_max_completion_length():
    long_response = "x" * 5000
    model = StubModel(responses=[long_response])
    result = generate_group(model, "prompt", group_size=4, max_new_tokens=100)
    assert len(result) == 4
    assert all(len(c) <= 100 for c in result)


def test_temperature_zero_produces_deterministic_output():
    model = StubModel()
    result = generate_group(model, "prompt", group_size=8, temperature=0.0)
    assert len(result) == 8
    assert len(set(result)) == 1
    assert result[0] == "deterministic-response"


def test_handles_empty_or_degenerate_prompt_gracefully():
    model = StubModel()
    result_empty = generate_group(model, "", group_size=4)
    assert len(result_empty) == 4
    assert all(isinstance(r, str) for r in result_empty)

    model_ws = StubModel()
    result_ws = generate_group(model_ws, "   ", group_size=4)
    assert len(result_ws) == 4
    assert all(isinstance(r, str) for r in result_ws)

    raising = RaisingOnEmptyModel()
    result_raise = generate_group(raising, "", group_size=4)
    assert result_raise == ["", "", "", ""]


def test_group_size_zero_returns_empty():
    model = StubModel()
    result = generate_group(model, "prompt", group_size=0)
    assert result == []


def test_group_size_negative_returns_empty():
    model = StubModel()
    result = generate_group(model, "prompt", group_size=-3)
    assert result == []


def test_uses_model_generate_protocol():
    assert isinstance(StubModel(), GenerativeModel) is True

    class NonConforming:
        pass

    with pytest.raises((TypeError, AttributeError)):
        generate_group(NonConforming(), "prompt", group_size=4)


def test_nonzero_temperature_may_produce_varied_output():
    varied = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    model = StubModel(responses=varied)
    result = generate_group(model, "prompt", group_size=8, temperature=0.9)
    assert len(result) == 8
    assert len(set(result)) > 1
