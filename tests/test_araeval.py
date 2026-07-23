"""
Unit tests for AraEval benchmark evaluation suite.
"""

from __future__ import annotations

import pytest

from araeval.config import AraEvalConfig, ARAEVAL_DATASETS
from araeval.evaluators import evaluate_ifeval, evaluate_mcq, extract_final_answer, parse_mcq_choice
from araeval.loader import AraEvalSample, load_araeval_task, normalize_sample


def test_config_task_resolution():
    cfg = AraEvalConfig(tasks=["all"])
    assert len(cfg.get_task_list()) == len(ARAEVAL_DATASETS)

    cfg_specific = AraEvalConfig(tasks=["math", "truthfulqa"])
    resolved = cfg_specific.get_task_list()
    assert "ara_math" in resolved
    assert "ara_truthfulqa" in resolved


def test_sample_normalization():
    raw_mcq = {
        "id": "q1",
        "question": "ما هي عاصمة مصر؟",
        "options": {"A": "القاهرة", "B": "الإسكندرية", "C": "الجيزة", "D": "أسوان"},
        "answer": "A",
    }
    sample = normalize_sample(raw_mcq, "test_mcq", 0)
    assert sample.id == "q1"
    assert "القاهرة" in sample.prompt
    assert sample.options["A"] == "القاهرة"
    assert sample.gold_answer == "A"


def test_extract_final_answer_with_think_tags():
    completion = "<think>\nنحلل المسألة خطوة بخطوة...\nالناتج = 42\n</think>\n\n<answer>B</answer>"
    extracted = extract_final_answer(completion)
    assert extracted == "B"

    completion_no_answer = "<think>\nعملية التفكير\n</think>\nالخيارات الصحيحة هي الإجابة B."
    extracted_no_ans = extract_final_answer(completion_no_answer)
    assert "الإجابة B" in extracted_no_ans


def test_mcq_evaluation():
    options = {"A": "30", "B": "40", "C": "13", "D": "26"}

    # Direct letter response
    is_correct, pred = evaluate_mcq("B", "B", options)
    assert is_correct is True
    assert pred == "B"

    # CoT answer block response
    completion = "<think>8 * 5 = 40</think><answer>B</answer>"
    is_correct_cot, pred_cot = evaluate_mcq(completion, "B", options)
    assert is_correct_cot is True
    assert pred_cot == "B"

    # Text value match response
    completion_text = "<think>المساحة 40 سم2</think> الإجابة الصحيحة هي الخيار B (40)"
    is_correct_text, pred_text = evaluate_mcq(completion_text, "B", options)
    assert is_correct_text is True
    assert pred_text == "B"


def test_ifeval_instruction_evaluation():
    instructions = [
        {"type": "include_keyword", "keyword": "الخوارزميات"},
        {"type": "exclude_keyword", "keyword": "الحاسوب"},
    ]

    pass_text = "تطوير الخوارزميات الذكية ساهم في التقدم."
    strict_pass, ratio, _ = evaluate_ifeval(pass_text, instructions)
    assert strict_pass is True
    assert ratio == 1.0

    fail_text = "تطوير الخوارزميات عبر جهاز الحاسوب."
    strict_pass_fail, ratio_fail, _ = evaluate_ifeval(fail_text, instructions)
    assert strict_pass_fail is False
    assert ratio_fail == 0.5


def test_mock_loader():
    samples = load_araeval_task("ara_math", limit=1)
    assert len(samples) == 1
    assert samples[0].task_name == "ara_math"
    assert "8 سم" in samples[0].prompt
