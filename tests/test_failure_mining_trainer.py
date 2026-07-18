"""Production-grade unit tests for GRPOTrainerWithFailureMining internals.

Tests zero-variance detection, RL-ZVP all_wrong handling, POPO replay buffer
interaction, and CRPS trace recording/hint injection.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from rlvr.failure_mining import (
    ReplayBuffer,
    ReplayEntry,
    is_zero_variance_group,
    is_effective_group,
    replace_ineffective_group,
)
from rlvr.crps import (
    SuccessTrace,
    SuccessTraceStore,
    build_progressive_suffix,
    find_related_success_trace,
    get_crps_suffix_fraction,
)


def _make_trainer(zero_variance_strategy="direct_scoring", enable_crps=False):
    from rlvr_sota.failure_mining_trainer import GRPOTrainerWithFailureMining

    model = MagicMock()
    model.training = True
    model.parameters = lambda: [MagicMock(requires_grad=True)]

    with patch.multiple(
        "rlvr_sota.failure_mining_trainer.GRPOTrainer",
        __init__=MagicMock(return_value=None),
        create_optimizer_and_scheduler=MagicMock(),
    ):
        trainer = GRPOTrainerWithFailureMining.__new__(GRPOTrainerWithFailureMining)
        trainer.zero_variance_strategy = zero_variance_strategy
        trainer.replay_buffer = ReplayBuffer(max_size=512)
        trainer.enable_crps = enable_crps
        trainer.crps_max_age_steps = 100
        trainer.success_trace_store = SuccessTraceStore(max_traces=256)
        trainer._failure_mining_stats = {}
        trainer._failure_counter = {}
        trainer._curriculum_sampler = None
        trainer._curriculum_config = None
        trainer.model = model
        trainer.args = MagicMock(logging_steps=10, steps_per_generation=1)
        trainer.num_generations = 2
        trainer.reward_funcs = []
        trainer.reward_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        trainer.compute_loss = lambda *a, **kw: torch.tensor(0.0)
        trainer.log = MagicMock()
        trainer.log_metrics = MagicMock(return_value={})
        trainer.control = MagicMock()
        trainer._train_batch_size = 1
        trainer.use_vllm = False
        trainer.state = MagicMock(global_step=10)
        trainer.processing_class = MagicMock()
        trainer.processing_class.decode = lambda ids, skip_special_tokens=False: "test_token"
        return trainer


def _make_completion_ids(B=4, T=8):
    return torch.randint(0, 1000, (B, T))


# ----------------------------------------------------------------------
# RL-ZVP: all_wrong detection uses CORRECTNESS component
# ----------------------------------------------------------------------


def test_all_wrong_uses_correctness_reward():
    """BUG-1 fix: all_wrong must check correctness (col 0), not combined reward."""
    trainer = _make_trainer(enable_crps=False)

    B, T, num_gen = 4, 8, 2
    advantages = torch.zeros(B)
    completion_mask = torch.ones(B, T, dtype=torch.long)
    old_per_token_logps = torch.randn(B, T)

    rewards_per_func = torch.zeros(B, 4)
    rewards_per_func[0:2, 0] = 0.0
    rewards_per_func[0:2, 1] = 1.0
    rewards_per_func[0:2, 2] = 1.0
    rewards_per_func[2:4, 0] = 1.0
    rewards_per_func[2:4, 2] = 1.0

    rewards = (rewards_per_func * trainer.reward_weights.unsqueeze(0)).nansum(dim=1)

    output = {"advantages": advantages}

    trainer._process_failure_mining(
        output, rewards_per_func, rewards, advantages,
        completion_mask, old_per_token_logps,
        num_gen, B // num_gen,
    )

    result_adv = output["advantages"]
    assert result_adv.ndim == 1, "advantages must remain 1D (scalar per-completion)"
    assert (result_adv[0:2] <= 0).all(), "All-wrong group: non-positive scalar advantage"
    assert "effective_sample_ratio" in trainer._failure_mining_stats


def test_rlzvp_all_wrong_produces_negative_scalar():
    trainer = _make_trainer(enable_crps=False)

    B, T, num_gen = 4, 8, 2
    advantages = torch.zeros(B)
    completion_mask = torch.ones(B, T, dtype=torch.long)
    old_per_token_logps = torch.randn(B, T).abs() * 0.1 + 0.1

    rewards_per_func = torch.zeros(B, 4)
    rewards_per_func[:, 0] = 0.0
    rewards = (rewards_per_func * trainer.reward_weights.unsqueeze(0)).nansum(dim=1)

    output = {"advantages": advantages}
    trainer._process_failure_mining(
        output, rewards_per_func, rewards, advantages,
        completion_mask, old_per_token_logps,
        num_gen, B // num_gen,
    )

    result_adv = output["advantages"]
    assert (result_adv <= 0).all(), "All-correctness-zero -> all-wrong -> negative signal"


# ----------------------------------------------------------------------
# V1 library: is_zero_variance_group and is_effective_group
# ----------------------------------------------------------------------


def test_is_zero_variance_group_same_rewards():
    assert is_zero_variance_group([0.5, 0.5, 0.5, 0.5])
    assert is_zero_variance_group([1.0, 1.0, 1.0, 1.0])
    assert is_zero_variance_group([0.0, 0.0])


def test_is_zero_variance_group_different_rewards():
    assert not is_zero_variance_group([0.0, 0.5, 1.0])
    assert not is_zero_variance_group([0.0, 0.0001])


def test_is_effective_group():
    assert is_effective_group([0.0, 0.0, 1.0])
    assert is_effective_group([0.0, 0.5, 1.0])


def test_is_effective_group_zero_variance():
    assert not is_effective_group([0.0, 0.0, 0.0])
    assert not is_effective_group([1.0, 1.0, 1.0])


def test_replace_ineffective_group_returns_none_for_effective():
    buf = ReplayBuffer(max_size=10)
    assert replace_ineffective_group(buf, [0.0, 0.5, 1.0]) is None


def test_replace_ineffective_group_empty_buffer():
    buf = ReplayBuffer(max_size=10)
    assert replace_ineffective_group(buf, [0.0, 0.0, 0.0]) is None


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(max_size=10)
    entry = ReplayEntry(
        prompt_id="test", completions=[], rewards=[0.0, 0.5, 1.0],
        old_policy_log_probs=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        step_generated=1,
    )
    buf.push(entry)
    assert len(buf) == 1
    sampled = buf.sample()
    assert sampled is not None
    assert sampled.prompt_id == "test"


# ----------------------------------------------------------------------
# CRPS: trace recording with individual decoded tokens
# ----------------------------------------------------------------------


def test_crps_trace_records_decoded_word_list():
    trainer = _make_trainer(enable_crps=True)
    trainer.processing_class.decode = (
        lambda ids, skip_special_tokens=True: " ".join(str(i) for i in ids)
    )

    B, num_gen, num_groups = 4, 2, 2
    completion_ids = torch.tensor([
        [101, 102, 103],
        [201, 202, 203],
        [301, 302, 303],
        [401, 402, 403],
    ])
    # columns: correctness, format, ...
    rewards_per_func = torch.tensor([
        [1.0, 1.0],
        [0.5, 1.0],
        [0.0, 0.0],
        [1.0, 0.8],
    ])
    inputs = [
        {"difficulty_tag": "easy", "sample_id": "1"},
        {"difficulty_tag": "medium", "sample_id": "2"},
        {"difficulty_tag": "hard", "sample_id": "3"},
        {"difficulty_tag": "hard", "sample_id": "4"},
    ]

    trainer._maybe_record_success_traces(
        completion_ids, rewards_per_func, inputs, num_gen, num_groups
    )

    # One success per group (first perfect-correctness+format>0 completion).
    assert len(trainer.success_trace_store) == 2
    first_trace = trainer.success_trace_store.traces[0]
    assert isinstance(first_trace.token_sequence, list)
    assert first_trace.token_sequence == ["101", "102", "103"]
    assert first_trace.difficulty_tag == "easy"

    second_trace = trainer.success_trace_store.traces[1]
    assert second_trace.difficulty_tag == "hard"
    assert second_trace.token_sequence == ["401", "402", "403"]


def test_crps_hint_injects_suffix_into_prompt():
    trainer = _make_trainer(enable_crps=True)
    trainer.state.global_step = 50

    trace = SuccessTrace(
        prompt_id="test", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["Step", "1:", "x=5.", "Step", "2:", "answer", "8."],
        step_recorded=10,
    )
    trainer.success_trace_store.add(trace)

    inputs = [
        {
            "prompt": [{"role": "user", "content": "What is 5+3?"}],
            "domain": "math",
            "puzzle_type": "math",
            "difficulty_tag": "hard",
            "sample_id": "h1",
        },
        {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "domain": "math",
            "difficulty_tag": "easy",
            "sample_id": "e1",
        },
    ]

    # failure_count=0 → fraction 0 (no hint yet); second pass injects.
    trainer._inject_crps_hints(inputs)
    result = trainer._inject_crps_hints(inputs)

    assert "Hint (partial solution)" in result[0]["prompt"][-1]["content"]
    assert "Hint (partial solution)" not in result[1]["prompt"][-1]["content"]
    key = trainer._crps_failure_key(inputs[0])
    assert trainer._failure_counter.get(key, 0) >= 1


def test_crps_failure_counter_increments():
    trainer = _make_trainer(enable_crps=True)
    trace = SuccessTrace(
        prompt_id="t", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["A", "B", "C"], step_recorded=1,
    )
    trainer.success_trace_store.add(trace)

    inputs = [
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "hard", "sample_id": "a"},
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "hard", "sample_id": "b"},
    ]

    # Warm counters past the fraction=0 first attempt.
    trainer._inject_crps_hints(inputs)
    result = trainer._inject_crps_hints(inputs)
    assert "Hint" in result[0]["prompt"][-1]["content"]
    assert "Hint" in result[1]["prompt"][-1]["content"]
    assert trainer._failure_counter.get(trainer._crps_failure_key(inputs[0]), 0) >= 1
    assert trainer._failure_counter.get(trainer._crps_failure_key(inputs[1]), 0) >= 1


def test_crps_no_hints_when_no_traces():
    trainer = _make_trainer(enable_crps=True)
    inputs = [
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "hard"},
    ]
    result = trainer._inject_crps_hints(inputs)
    assert "Hint" not in result[0]["prompt"][-1]["content"]


def test_crps_hint_not_injected_for_easy_problems():
    trainer = _make_trainer(enable_crps=True)
    trace = SuccessTrace(
        prompt_id="test", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["reasoning..."], step_recorded=1,
    )
    trainer.success_trace_store.add(trace)

    inputs = [
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "easy"},
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "trivial"},
        {"prompt": [{"role": "user", "content": "Q?"}],
         "domain": "math", "difficulty_tag": "medium"},
    ]
    result = trainer._inject_crps_hints(inputs)
    for inp in result:
        assert "Hint" not in inp["prompt"][-1]["content"]


def test_crps_hint_deep_copies_input():
    trainer = _make_trainer(enable_crps=True)
    trace = SuccessTrace(
        prompt_id="t", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["A", "B", "C"], step_recorded=1,
    )
    trainer.success_trace_store.add(trace)

    original_prompt = [{"role": "user", "content": "Q?"}]
    inputs = [
        {"prompt": original_prompt, "domain": "math", "difficulty_tag": "hard", "sample_id": "x"},
    ]

    # First call only bumps failure_count (fraction=0); second injects a hint.
    trainer._inject_crps_hints(inputs)
    result = trainer._inject_crps_hints(inputs)

    assert inputs[0]["prompt"] is original_prompt
    assert inputs[0]["prompt"][-1]["content"] == "Q?"
    assert result[0] is not inputs[0]
    assert "Hint" in result[0]["prompt"][-1]["content"]


# ----------------------------------------------------------------------
# CRPS v1 library: suffix building
# ----------------------------------------------------------------------


def test_build_progressive_suffix_small_fraction():
    trace = SuccessTrace(
        prompt_id="t", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        step_recorded=1,
    )
    result = build_progressive_suffix(trace, 0.3)
    assert result == "H I J", f"Expected last 3 tokens, got: {result}"


def test_build_progressive_suffix_full_trace():
    trace = SuccessTrace(
        prompt_id="t", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["X", "Y", "Z"],
        step_recorded=1,
    )
    result = build_progressive_suffix(trace, 1.0)
    assert result == "X Y Z"


def test_get_crps_suffix_fraction_schedule():
    assert get_crps_suffix_fraction(0) == 0.0
    assert get_crps_suffix_fraction(1) == 0.1
    assert get_crps_suffix_fraction(2) == 0.3
    assert get_crps_suffix_fraction(3) == 0.5
    assert get_crps_suffix_fraction(5) == 0.5


def test_find_related_success_trace():
    store = SuccessTraceStore(max_traces=10)
    store.add(SuccessTrace(
        prompt_id="p1", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["reasoning..."], step_recorded=5,
    ))
    store.add(SuccessTrace(
        prompt_id="p2", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["more..."], step_recorded=15,
    ))

    result = find_related_success_trace(
        store.traces, puzzle_type="math", difficulty_tag="hard",
        max_age_steps=100, current_step=20,
    )
    assert result is not None
    assert result.prompt_id == "p2", "Should return most recent matching trace"


def test_find_related_success_trace_no_match():
    store = SuccessTraceStore(max_traces=10)
    store.add(SuccessTrace(
        prompt_id="p1", puzzle_type="logic", difficulty_tag="hard",
        token_sequence=["r"], step_recorded=1,
    ))
    result = find_related_success_trace(
        store.traces, puzzle_type="math", difficulty_tag="hard",
        max_age_steps=100, current_step=20,
    )
    assert result is None


def test_find_related_success_trace_stale():
    store = SuccessTraceStore(max_traces=10)
    store.add(SuccessTrace(
        prompt_id="p1", puzzle_type="math", difficulty_tag="hard",
        token_sequence=["r"], step_recorded=5,
    ))
    result = find_related_success_trace(
        store.traces, puzzle_type="math", difficulty_tag="hard",
        max_age_steps=10, current_step=100,
    )
    assert result is None
