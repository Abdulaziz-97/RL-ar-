"""Tests for the TRL-compatible reward functions."""

from rlvr_pipeline.rewards import (
    ALL_REWARD_FUNCS,
    DEFAULT_REWARD_WEIGHTS,
    answer_leak_penalty_func,
    composite_reward_func,
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    length_penalty_func,
    structural_leak_penalty_func,
)

ARABIC_REASONING = (
    "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u0645\u0639\u0627 "
    "\u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629 \u062c\u062f\u0627</think><answer>4</answer>"
)

HACK_EMPTY = "<think></think><answer>999</answer>"


def _conv(text):
    return [{"role": "assistant", "content": text}]


def test_correctness_reward_func_conversational():
    completions = [_conv(ARABIC_REASONING), _conv("<think>x</think><answer>5</answer>")]
    rewards = correctness_reward_func(
        prompts=["p1", "p2"],
        completions=completions,
        ground_truth_answer=[4, 5],
        domain=["math", "math"],
    )
    assert rewards == [1.0, 1.0]


def test_correctness_reward_func_string_completions():
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=["<think>r</think><answer>16.0</answer>"],
        ground_truth_answer=[16],
        domain=["math"],
    )
    assert rewards == [1.0]


def test_correctness_reward_func_logic_domain():
    completions = [_conv('<think>r</think><answer>{"A": 1, "B": 2}</answer>')]
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=completions,
        ground_truth_answer=[{"A": 1, "B": 2}],
        domain=["logic"],
        puzzle_type=["constraint"],
    )
    assert rewards == [1.0]


def test_correctness_reward_func_logic_gt_json_string():
    """Loader serializes dict GTs to JSON strings — must still match."""
    import json

    gt = {"A": 1, "B": 2}
    ans = json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=[_conv(f"<think>enough reasoning tokens here now yes</think><answer>{ans}</answer>")],
        ground_truth_answer=[json.dumps(gt, ensure_ascii=False)],
        domain=["logic"],
    )
    assert rewards == [1.0]


def test_format_reward_func():
    good = _conv(ARABIC_REASONING)
    bad = _conv(HACK_EMPTY)
    rewards = format_reward_func(prompts=["p1", "p2"], completions=[good, bad])
    assert rewards[0] > 0.0
    # Empty <think></think> earns nothing; non-empty answer alone ≤ 0.13
    assert rewards[1] <= 0.13
    assert rewards[0] > rewards[1]


def test_format_reward_partial_think_only():
    """Non-empty think without answer gets small partial credit only."""
    text = (
        "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 "
        "\u0645\u0639\u0627 \u062b\u0645 \u0623\u062d\u0633\u0628</think>"
    )
    rewards = format_reward_func(prompts=["p1"], completions=[_conv(text)])
    assert 0.08 <= rewards[0] <= 0.15
    # Far below a full format success.
    assert rewards[0] < 0.5


def test_format_reward_empty_tags_earn_zero():
    assert format_reward_func(prompts=["p"], completions=[_conv("<think></think>")])[0] == 0.0
    assert format_reward_func(prompts=["p"], completions=[_conv("<answer></answer>")])[0] == 0.0


def test_format_partial_never_beats_full():
    full = format_reward_func(prompts=["p"], completions=[_conv(ARABIC_REASONING)])[0]
    # Strict-failing partial: think only (no answer tags) — must stay << full.
    partial_only = (
        "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 "
        "\u0645\u0639\u0627 \u062b\u0645 \u0623\u062d\u0633\u0628</think>"
    )
    spam = format_reward_func(prompts=["p"], completions=[_conv(partial_only)])[0]
    assert full > spam
    assert spam <= 0.30
    # Empty tags must not pay.
    assert format_reward_func(prompts=["p"], completions=[_conv("<think></think><answer></answer>")])[0] == 0.0


def test_language_reward_func_arabic():
    rewards = language_reward_func(
        prompts=["p1"], completions=[_conv(ARABIC_REASONING)]
    )
    assert rewards[0] > 0.8


def test_language_reward_ungated_without_answer_close():
    """Language must score Arabic think text even without </answer>."""
    text = (
        "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 "
        "\u0645\u0639\u0627 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c</think><answer>4"
    )
    rewards = language_reward_func(prompts=["p1"], completions=[_conv(text)])
    assert rewards[0] > 0.8


def test_language_reward_func_english():
    rewards = language_reward_func(
        prompts=["p1"],
        completions=[_conv("<think>First I add then I compute carefully here</think><answer>4</answer>")],
    )
    assert rewards[0] < 0.3


def test_answer_leak_penalty_func_triggers():
    leak = _conv("<think>the answer is the answer is the answer is the answer is</think><answer>42</answer>")
    rewards = answer_leak_penalty_func(prompts=["p1"], completions=[leak])
    assert rewards[0] == -1.0


def test_answer_leak_penalty_func_clean():
    rewards = answer_leak_penalty_func(
        prompts=["p1"], completions=[_conv(ARABIC_REASONING)]
    )
    assert rewards[0] == 0.0


def test_structural_leak_penalty_func_triggers():
    bad = _conv("First I add two and two to get four. <think></think><answer>4</answer>")
    rewards = structural_leak_penalty_func(prompts=["p1"], completions=[bad])
    assert rewards[0] == -1.0


def test_structural_leak_penalty_func_clean():
    rewards = structural_leak_penalty_func(
        prompts=["p1"], completions=[_conv(ARABIC_REASONING)]
    )
    assert rewards[0] == 0.0


def test_all_reward_funcs_count_and_weights():
    assert len(ALL_REWARD_FUNCS) == 6
    assert len(DEFAULT_REWARD_WEIGHTS) == 6
    assert DEFAULT_REWARD_WEIGHTS == [0.6, 0.2, 0.05, 0.5, 0.3, 0.15]


def test_length_penalty_func_triggers_on_long_think():
    long_think = " ".join(["كلمة"] * 200)
    completions = [_conv(f"<think>{long_think}</think><answer>4</answer>")]
    rewards = length_penalty_func(prompts=["p1"], completions=completions)
    assert rewards[0] == -1.0


def test_length_penalty_func_clean_short_think():
    rewards = length_penalty_func(
        prompts=["p1"], completions=[_conv(ARABIC_REASONING)]
    )
    assert rewards[0] == 0.0


def test_composite_reward_func_matches_prd_formula():
    rewards = composite_reward_func(
        prompts=["p1", "p2"],
        completions=[_conv(ARABIC_REASONING), _conv(HACK_EMPTY)],
        ground_truth_answer=[4, 998],
        domain=["math", "math"],
    )
    assert rewards[0] >= 0.9
    assert rewards[1] <= 0.2


def test_reward_funcs_handle_missing_columns():
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=["<think>r</think><answer>4</answer>"],
        ground_truth_answer=[4],
        domain=["math"],
    )
    assert len(rewards) == 1
    assert rewards[0] == 1.0


def test_malformed_nonempty_answer_spec_fails_closed():
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=["<think>أحسب الناتج خطوة خطوة ثم أتحقق منه جيدًا</think><answer>4</answer>"],
        ground_truth_answer=[4],
        domain=["math"],
        answer_spec=["not-json"],
    )
    assert rewards == [0.0]


def test_logic_boolean_is_not_equal_to_integer():
    spec = '{"type":"logic_json","canonical":{"x":true}}'
    rewards = correctness_reward_func(
        prompts=["p1"],
        completions=['<think>أطبق القيود ثم أتحقق من كل قيمة بدقة</think><answer>{"x":1}</answer>'],
        ground_truth_answer=['{"x":true}'],
        domain=["logic"],
        answer_spec=[spec],
    )
    assert rewards == [0.0]
