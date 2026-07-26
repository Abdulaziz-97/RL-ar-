"""
Unit tests for English baseline evaluation suite.
"""

import pytest
from english_eval.config import EnglishEvalConfig, DEFAULT_ENGLISH_TASKS
from english_eval.evaluators import (
    evaluate_gsm8k,
    evaluate_ifeval_item,
    evaluate_mcq,
    extract_final_answer,
    parse_mcq_choice,
)


def test_extract_final_answer():
    text_with_think = "<think>Some reasoning here</think>\n\n<answer>Option B</answer>"
    assert extract_final_answer(text_with_think) == "Option B"

    text_no_answer_tag = "<think>Some thinking</think>Final result is 42"
    assert extract_final_answer(text_no_answer_tag) == "Final result is 42"


def test_parse_mcq_choice():
    options = {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"}

    assert parse_mcq_choice("A", options) == "A"
    assert parse_mcq_choice("The correct answer is B.", options) == "B"
    assert parse_mcq_choice("Option C: Third option", options) == "C"


def test_evaluate_mcq():
    options = {"A": "Cat", "B": "Dog", "C": "Bird", "D": "Fish"}
    is_correct, pred = evaluate_mcq("Answer: B", "B", options)
    assert is_correct is True
    assert pred == "B"

    is_correct_wrong, pred_wrong = evaluate_mcq("Answer: A", "B", options)
    assert is_correct_wrong is False
    assert pred_wrong == "A"


def test_evaluate_gsm8k():
    completion = "Let's calculate step by step:\n3 + 4 = 7\n#### 7"
    is_correct, pred, gold = evaluate_gsm8k(completion, "The answer is #### 7")
    assert is_correct is True
    assert pred == "7"
    assert gold == "7"


def test_evaluate_ifeval_item():
    completion = "This is a response containing keyword apple.\n\nParagraph two."
    inst_ids = ["include_keyword", "number_paragraphs"]
    kwargs = [{"keywords": ["apple"]}, {"num_paragraphs": 2}]

    all_passed, score, details = evaluate_ifeval_item(completion, inst_ids, kwargs)
    assert all_passed is True
    assert score == 1.0


def test_english_eval_config():
    cfg = EnglishEvalConfig(tasks=["all"])
    assert cfg.get_task_list() == DEFAULT_ENGLISH_TASKS

    cfg_specific = EnglishEvalConfig(tasks=["ifeval", "gsm8k"])
    assert cfg_specific.get_task_list() == ["ifeval", "gsm8k"]
