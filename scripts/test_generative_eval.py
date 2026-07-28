"""TDD Test Suite for Generation-Based Arabic Evaluation.

Written BEFORE the implementation code.
Every test defines a contract that the implementation MUST satisfy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "official_eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_tests() -> None:
    print("=================================================================")
    print("TDD TEST SUITE: GENERATION-BASED ARABIC EVALUATION ENGINE")
    print("=================================================================\n")

    # ------------------------------------------------------------------
    # Test 1: Thinking Tag Stripping
    # ------------------------------------------------------------------
    print("[Test 1] Thinking Tag Stripping...")
    from tasks.araeval.generative_utils import strip_thinking_tags

    # Basic thinking tags
    assert strip_thinking_tags("<think>Let me analyze this...</think>The answer is A") == "The answer is A"
    # Nested content inside thinking
    assert strip_thinking_tags("<think>\nStep 1: read\nStep 2: solve\n</think>\nA") == "\nA"
    # No thinking tags
    assert strip_thinking_tags("The answer is B") == "The answer is B"
    # Empty thinking
    assert strip_thinking_tags("<think></think>C") == "C"
    # Multiple thinking blocks (should strip all)
    assert strip_thinking_tags("<think>first</think>middle<think>second</think>end") == "middleend"
    print("  --> PASS: Thinking tag stripping works for all edge cases!\n")

    # ------------------------------------------------------------------
    # Test 2: Answer Extraction from Generated Text
    # ------------------------------------------------------------------
    print("[Test 2] Answer Extraction from Generated Arabic/Latin Text...")
    from tasks.araeval.generative_utils import extract_answer

    # Latin letter answers (what AraEval actually uses)
    assert extract_answer("الإجابة: A") == "A"
    assert extract_answer("الإجابة هي: B") == "B"
    assert extract_answer("الإجابة: C.") == "C"
    assert extract_answer("The answer is D") == "D"

    # Answer at end of text
    assert extract_answer("After careful analysis, the correct option is B") == "B"
    assert extract_answer("بعد التحليل، الإجابة الصحيحة هي A") == "A"

    # Answer after thinking tags (stripped before extraction)
    assert extract_answer("<think>Let me think step by step...</think>\nالإجابة: C") == "C"

    # Answer with surrounding whitespace
    assert extract_answer("  A  ") == "A"
    assert extract_answer("\nB\n") == "B"

    # Just a single letter
    assert extract_answer("A") == "A"
    assert extract_answer("D") == "D"

    # Extraction failure (no valid answer)
    assert extract_answer("I don't know the answer") is None
    assert extract_answer("") is None

    # Answer embedded in longer reasoning
    assert extract_answer("الخيار الصحيح هو C لأن...") == "C"

    # Multiple letters - should pick the last one (final answer)
    assert extract_answer("Option A is wrong. Option B is also wrong. The answer is C") == "C"

    print("  --> PASS: Answer extraction works for all edge cases!\n")

    # ------------------------------------------------------------------
    # Test 3: Answer Grading
    # ------------------------------------------------------------------
    print("[Test 3] Answer Grading Logic...")
    from tasks.araeval.generative_utils import grade_answer

    # Correct answers
    assert grade_answer("A", 0) is True   # A = index 0
    assert grade_answer("B", 1) is True   # B = index 1
    assert grade_answer("C", 2) is True   # C = index 2
    assert grade_answer("D", 3) is True   # D = index 3

    # Incorrect answers
    assert grade_answer("A", 1) is False
    assert grade_answer("B", 0) is False
    assert grade_answer("C", 3) is False

    # Extraction failure (None) is always incorrect
    assert grade_answer(None, 0) is False
    assert grade_answer(None, 3) is False

    # Case insensitive
    assert grade_answer("a", 0) is True
    assert grade_answer("b", 1) is True

    print("  --> PASS: Grading logic exact for correct, incorrect, and None cases!\n")

    # ------------------------------------------------------------------
    # Test 4: Generative MCQ Prompt Building
    # ------------------------------------------------------------------
    print("[Test 4] Generative MCQ Prompt Building...")
    from tasks.araeval.generative_utils import build_generative_prompt

    prompt = build_generative_prompt(
        question="ما هو أكبر كوكب في المجموعة الشمسية؟",
        choices=["المريخ", "المشتري", "زحل", "الأرض"],
    )
    # Must contain the question
    assert "ما هو أكبر كوكب" in prompt
    # Must contain all choices with labels
    assert "A." in prompt
    assert "B." in prompt
    assert "C." in prompt
    assert "D." in prompt
    assert "المريخ" in prompt
    assert "المشتري" in prompt
    assert "زحل" in prompt
    assert "الأرض" in prompt
    # Must end with answer instruction
    assert prompt.strip().endswith("الإجابة:")

    # With context/passage
    prompt_ctx = build_generative_prompt(
        question="ما هي النتيجة؟",
        choices=["10", "20"],
        context="في تجربة علمية...",
    )
    assert "السياق:" in prompt_ctx
    assert "في تجربة علمية" in prompt_ctx

    print("  --> PASS: Generative MCQ prompt builder works perfectly!\n")

    # ------------------------------------------------------------------
    # Test 5: Per-Task Generation Hyperparameters
    # ------------------------------------------------------------------
    print("[Test 5] Per-Task Generation Hyperparameters...")
    from tasks.araeval.generative_utils import GENERATION_PROFILES, get_generation_profile

    # All profiles must have at least 768 tokens for reasoning
    mcq_profile = get_generation_profile("araeval_ien_mcq")
    assert mcq_profile["max_new_tokens"] >= 768, f"MCQ max_new_tokens should be >= 768: {mcq_profile['max_new_tokens']}"
    assert mcq_profile["temperature"] == 0.0, "MCQ temperature should be 0.0 (greedy)"

    # Math should allow at least 768 tokens reasoning
    math_profile = get_generation_profile("araeval_aramath")
    assert math_profile["max_new_tokens"] >= 768, f"Math max_new_tokens should be >= 768: {math_profile['max_new_tokens']}"

    # AraPro (long passages) should have conservative batch_size
    arapro_profile = get_generation_profile("araeval_arapro")
    assert arapro_profile["batch_size"] <= 8, f"AraPro batch_size too high: {arapro_profile['batch_size']}"

    # IFEval should have longest max_new_tokens
    ifeval_profile = get_generation_profile("araeval_ifeval")
    assert ifeval_profile["max_new_tokens"] >= 1024, f"IFEval max_new_tokens too low: {ifeval_profile['max_new_tokens']}"

    # Unknown task falls back to default
    default_profile = get_generation_profile("unknown_task")
    assert default_profile is not None

    print("  --> PASS: Per-task generation hyperparameters verified!\n")

    # ------------------------------------------------------------------
    # Test 6: Checkpoint Save/Load/Resume
    # ------------------------------------------------------------------
    print("[Test 6] Checkpoint Save/Load/Resume...")
    from tasks.araeval.generative_utils import (
        new_generative_checkpoint,
        load_generative_checkpoint,
        save_generative_checkpoint,
        pending_generative_tasks,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint.json"

        # New checkpoint
        ckpt = new_generative_checkpoint(
            model="unsloth/Qwen3.5-4B",
            adapter_path=None,
            enable_thinking=False,
        )
        assert ckpt["version"] == 1
        assert ckpt["evaluation_mode"] == "generative"
        assert ckpt["completed"] == {}

        # Save checkpoint
        save_generative_checkpoint(ckpt_path, ckpt)
        assert ckpt_path.is_file()

        # Load checkpoint
        loaded = load_generative_checkpoint(ckpt_path)
        assert loaded["version"] == 1
        assert loaded["model"] == "unsloth/Qwen3.5-4B"

        # Pending tasks (all 6 MCQ tasks should be pending; IFEval is excluded)
        from tasks.araeval.generative_utils import GENERATIVE_TASKS
        pending = pending_generative_tasks(loaded)
        assert len(pending) == len(GENERATIVE_TASKS)

        # Mark one task complete
        loaded["completed"]["araeval_ien_mcq"] = {
            "correct": 7000,
            "total": 9990,
            "accuracy": 70.07,
            "extraction_failures": 5,
            "seconds": 120.5,
        }
        save_generative_checkpoint(ckpt_path, loaded)
        reloaded = load_generative_checkpoint(ckpt_path)
        pending2 = pending_generative_tasks(reloaded)
        assert len(pending2) == len(GENERATIVE_TASKS) - 1
        assert "araeval_ien_mcq" not in pending2

    print("  --> PASS: Checkpoint save/load/resume works perfectly!\n")

    # ------------------------------------------------------------------
    # Test 7: Summary Report Generation
    # ------------------------------------------------------------------
    print("[Test 7] Summary Report Generation...")
    from tasks.araeval.generative_utils import summarize_generative_checkpoint

    mock_ckpt = new_generative_checkpoint(
        model="unsloth/Qwen3.5-4B",
        adapter_path="aziz9788/qwen3.5-4b-arabic-grpo-v2",
        enable_thinking=True,
    )
    mock_ckpt["completed"] = {
        "araeval_ien_mcq": {"correct": 7000, "total": 9990, "accuracy": 70.07, "extraction_failures": 5, "seconds": 120.0},
        "araeval_ien_tf": {"correct": 4000, "total": 5823, "accuracy": 68.73, "extraction_failures": 2, "seconds": 80.0},
        "araeval_aramath": {"correct": 400, "total": 605, "accuracy": 66.12, "extraction_failures": 1, "seconds": 60.0},
        "araeval_etec": {"correct": 1200, "total": 1887, "accuracy": 63.59, "extraction_failures": 0, "seconds": 40.0},
        "araeval_arapro": {"correct": 3500, "total": 5001, "accuracy": 69.99, "extraction_failures": 3, "seconds": 100.0},
        "araeval_truthfulqa": {"correct": 350, "total": 536, "accuracy": 65.30, "extraction_failures": 1, "seconds": 30.0},
    }

    summary = summarize_generative_checkpoint(mock_ckpt)
    assert summary["evaluation_mode"] == "generative"
    assert summary["enable_thinking"] is True
    assert summary["adapter_path"] == "aziz9788/qwen3.5-4b-arabic-grpo-v2"
    assert len(summary["task_results"]) == 6
    assert summary["task_results"]["araeval_ien_mcq"]["accuracy"] == 70.07
    assert isinstance(summary["overall_generative_accuracy"], float)
    assert summary["overall_generative_accuracy"] > 0
    assert isinstance(summary["total_extraction_failures"], int)
    assert summary["total_extraction_failures"] == 12  # 5+2+1+0+3+1
    assert isinstance(summary["total_seconds"], float)
    assert summary["total_seconds"] == 430.0  # 120+80+60+40+100+30

    print("  --> PASS: Summary report generation verified!\n")

    # ------------------------------------------------------------------
    # Test 8: TruthfulQA Special Handling
    # ------------------------------------------------------------------
    print("[Test 8] TruthfulQA Special Prompt Handling...")
    # TruthfulQA has a different schema: instruction + input, choices are full text strings
    # The generative prompt should handle this correctly
    prompt_tqa = build_generative_prompt(
        question="ماذا يحدث إذا أكلت سمكا وشربت حليبا؟",
        choices=[
            "لا يحدث شيء إذا أكلت سمكًا وشربت حليبًا",
            "ستصاب بالبهاق",
            "ستصاب بتسمم معوي",
            "ستشعر بالغثيان",
        ],
        instruction="أجب على السؤال التالي :",
    )
    assert "أجب على السؤال التالي" in prompt_tqa
    assert "ماذا يحدث" in prompt_tqa
    assert "A." in prompt_tqa
    assert "D." in prompt_tqa

    print("  --> PASS: TruthfulQA prompt format verified!\n")

    print("=================================================================")
    print("ALL 8 TDD TESTS PASSED SUCCESSFULLY! 🏆")
    print("=================================================================")


if __name__ == "__main__":
    run_tests()
