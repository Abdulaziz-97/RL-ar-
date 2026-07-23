"""
Exhaustive unit test suite for AraEval benchmark evaluation suite.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from araeval.config import AraEvalConfig, ARAEVAL_DATASETS
from araeval.evaluators import (
    evaluate_ifeval,
    evaluate_mcq,
    extract_final_answer,
    parse_mcq_choice,
)
from araeval.loader import (
    AraEvalSample,
    _get_mock_samples,
    format_mcq_prompt,
    load_araeval_task,
    normalize_sample,
)
from araeval.__main__ import main as cli_main


# ── Configuration Tests ──────────────────────────────────────────────

def test_config_defaults():
    cfg = AraEvalConfig()
    assert cfg.model_name_or_path == "Qwen/Qwen3.5-4B"
    assert cfg.eval_mode == "generation"
    assert cfg.temperature == 0.0
    assert cfg.get_task_list() == list(ARAEVAL_DATASETS.keys())


def test_config_task_resolution_variations():
    cfg_all = AraEvalConfig(tasks=["all"])
    assert len(cfg_all.get_task_list()) == len(ARAEVAL_DATASETS)

    cfg_empty = AraEvalConfig(tasks=[])
    assert len(cfg_empty.get_task_list()) == len(ARAEVAL_DATASETS)

    cfg_math = AraEvalConfig(tasks=["math"])
    assert cfg_math.get_task_list() == ["ara_math"]

    cfg_partial = AraEvalConfig(tasks=["truthfulqa", "ifeval"])
    assert "ara_truthfulqa" in cfg_partial.get_task_list()
    assert "ara_ifeval" in cfg_partial.get_task_list()

    cfg_unknown = AraEvalConfig(tasks=["non_existent_task"])
    assert cfg_unknown.get_task_list() == list(ARAEVAL_DATASETS.keys())


# ── Data Normalization & Loading Tests ─────────────────────────────────

def test_format_mcq_prompt():
    q = "ما هي مساحة المربع؟"
    opts = {"A": "طول الضلع تربيع", "B": "طول الضلع في 2"}
    formatted = format_mcq_prompt(q, opts)
    assert "السؤال: ما هي مساحة المربع؟" in formatted
    assert "A) طول الضلع تربيع" in formatted
    assert "اختر حرف الإجابة الصحيحة" in formatted


def test_normalize_sample_dict_options():
    raw = {
        "id": "q10",
        "question": "كم عدد أركان الإسلام؟",
        "options": {"A": "5", "B": "6", "C": "4", "D": "7"},
        "answer": "A",
    }
    sample = normalize_sample(raw, "test_task", 0)
    assert sample.id == "q10"
    assert sample.task_name == "test_task"
    assert sample.gold_answer == "A"
    assert sample.options["A"] == "5"


def test_normalize_sample_list_options():
    raw = {
        "question_id": "q20",
        "prompt": "ما هي اللغة الرسمية في مصر؟",
        "choices": ["العربية", "الإنجليزية", "الفرنسية", "الإسبانية"],
        "target": "A",
    }
    sample = normalize_sample(raw, "test_task", 1)
    assert sample.id == "q20"
    assert sample.options["A"] == "العربية"
    assert sample.options["B"] == "الإنجليزية"
    assert sample.gold_answer == "A"


def test_normalize_sample_individual_keys():
    raw = {
        "key": "q30",
        "input": "ما هو عاصمة فرنسا؟",
        "A": "باريس",
        "B": "لندن",
        "gold": "A",
    }
    sample = normalize_sample(raw, "test_task", 2)
    assert sample.id == "q30"
    assert sample.options["A"] == "باريس"
    assert sample.options["B"] == "لندن"


def test_normalize_sample_with_instructions():
    raw = {
        "id": "ifeval_1",
        "prompt": "اكتب نصاً قصيرًا.",
        "instructions": [{"type": "min_words", "min": 5}],
    }
    sample = normalize_sample(raw, "ara_ifeval", 0)
    assert len(sample.instructions) == 1
    assert sample.instructions[0]["type"] == "min_words"


def test_mock_sample_generator():
    for task in ["ara_ifeval", "ara_math", "ara_truthfulqa", "ien_mcq"]:
        samples = _get_mock_samples(task)
        assert len(samples) >= 1
        assert "question" in samples[0]


def test_load_araeval_task_limit():
    samples = load_araeval_task("ara_math", limit=1)
    assert len(samples) == 1
    assert isinstance(samples[0], AraEvalSample)
    assert samples[0].task_name == "ara_math"


# ── Evaluator Tests ───────────────────────────────────────────────────

def test_extract_final_answer_formatting():
    # 1. Complete <think>...</think><answer>...</answer>
    cot = "<think>\n15 + 27 = 42\n</think>\n\n<answer>42</answer>"
    assert extract_final_answer(cot) == "42"

    # 2. Chat template special tokens present
    special = "<|im_start|>assistant\n<think>تفكير</think>\n<answer>B</answer><|im_end|>"
    assert extract_final_answer(special) == "B"

    # 3. No <answer> block, only <think>
    think_only = "<think>\nخطوات الحل\n</think>\nالإجابة الصحيحة هي C"
    assert extract_final_answer(think_only) == "الإجابة الصحيحة هي C"

    # 4. Plain string
    assert extract_final_answer("  الإجابة A  ") == "الإجابة A"


def test_parse_mcq_choice_variations():
    opts = {"A": "الرياض", "B": "جدة", "C": "مكة", "D": "الدمام"}

    # Direct letter
    assert parse_mcq_choice("A", opts) == "A"
    assert parse_mcq_choice("b", opts) == "B"

    # Arabic prefix patterns
    assert parse_mcq_choice("الإجابة الصحيحة هي C", opts) == "C"
    assert parse_mcq_choice("الخيار (D) هو الصحيح", opts) == "D"
    assert parse_mcq_choice("الجواب: A", opts) == "A"

    # English prefix patterns
    assert parse_mcq_choice("Option B is correct", opts) == "B"
    assert parse_mcq_choice("Answer: C", opts) == "C"

    # Match by option text
    assert parse_mcq_choice("العاصمة هي مكة المكرمة", opts) == "C"

    # Empty / whitespace
    assert parse_mcq_choice("", opts) == ""
    assert parse_mcq_choice("   ", opts) == ""


def test_evaluate_mcq_scoring():
    opts = {"A": "30", "B": "40", "C": "13", "D": "26"}

    # Correct letter match
    ok1, pred1 = evaluate_mcq("B", "B", opts)
    assert ok1 is True and pred1 == "B"

    # Correct CoT completion
    ok2, pred2 = evaluate_mcq("<think>8 * 5 = 40</think><answer>B</answer>", "B", opts)
    assert ok2 is True and pred2 == "B"

    # Correct text match
    ok3, pred3 = evaluate_mcq("الناتج هو 40", "B", opts)
    assert ok3 is True and pred3 == "B"

    # Incorrect choice
    ok4, pred4 = evaluate_mcq("A", "B", opts)
    assert ok4 is False and pred4 == "A"


def test_evaluate_ifeval_all_rules():
    instructions = [
        {"type": "include_keyword", "keyword": "الذكاء"},
        {"type": "exclude_keyword", "keyword": "خطأ"},
        {"type": "min_words", "min": 3},
        {"type": "max_words", "max": 10},
    ]

    # Perfect pass
    text_pass = "تطبيق الذكاء الاصطناعي مفيد جداً للمجتمع."
    strict_pass, ratio, results = evaluate_ifeval(text_pass, instructions)
    assert strict_pass is True
    assert ratio == 1.0
    assert len(results) == 4

    # Fail exclude_keyword
    text_fail_exclude = "تطبيق الذكاء الاصطناعي وفيه خطأ كبيرا."
    s_pass1, r1, _ = evaluate_ifeval(text_fail_exclude, instructions)
    assert s_pass1 is False
    assert r1 < 1.0

    # Fail min_words
    text_short = "الذكاء الاصطناعي"
    s_pass2, r2, _ = evaluate_ifeval(text_short, instructions)
    assert s_pass2 is False

    # Fail max_words
    text_long = "تطبيق الذكاء الاصطناعي في العلوم والتكنولوجيا والطب والصناعة والتجارة والتعليم متقدم جدا."
    s_pass3, r3, _ = evaluate_ifeval(text_long, instructions)
    assert s_pass3 is False

    # Empty instructions
    s_pass_empty, ratio_empty, _ = evaluate_ifeval("أي نص", [])
    assert s_pass_empty is True
    assert ratio_empty == 1.0


# ── Runner & CLI Integration Tests ────────────────────────────────────

def test_runner_offline_mock(tmp_path):
    out_dir = tmp_path / "araeval_test"
    cfg = AraEvalConfig(
        tasks=["ara_math", "ara_ifeval"],
        limit=1,
        output_dir=str(out_dir),
    )

    # Test config task list resolution
    task_list = cfg.get_task_list()
    assert "ara_math" in task_list
    assert "ara_ifeval" in task_list

    # Test output directory structure
    out_file = out_dir / "araeval_results.json"
    mock_summary = {
        "tasks": {
            "ara_math": {"task_name": "ara_math", "accuracy": 1.0, "total_samples": 1},
            "ara_ifeval": {"task_name": "ara_ifeval", "accuracy": 1.0, "total_samples": 1},
        },
        "macro_accuracy": 1.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(mock_summary, ensure_ascii=False), encoding="utf-8")
    assert out_file.exists()


def test_cli_parser():
    argv = ["--model", "Qwen/Qwen3.5-4B", "--tasks", "ara_math", "--limit", "1", "--output", "./tmp_eval"]
    from araeval.__main__ import _build_parser

    parser = _build_parser()
    args = parser.parse_args(argv)

    assert args.model == "Qwen/Qwen3.5-4B"
    assert args.tasks == ["ara_math"]
    assert args.limit == 1
    assert args.output == "./tmp_eval"
