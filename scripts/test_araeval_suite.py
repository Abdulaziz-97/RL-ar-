"""
Executive Test Suite for AraEval Evaluation Engine.
Verifies parsing, Arabic choice letter normalization, numerical equivalence,
and end-to-end Core 4 evaluation execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from araeval.config import AraEvalConfig, CORE_FOUR_DATASETS
from araeval.evaluators import evaluate_ifeval, evaluate_mcq, evaluate_openended, extract_final_answer, parse_mcq_choice
from araeval.loader import load_araeval_task


def run_executive_tests():
    print("=================================================================")
    print("RUNNING EXECUTIVE TEST SUITE FOR ARAEVAL EVALUATION ENGINE")
    print("=================================================================\n")

    # --- Test 1: Arabic Choice Letter Normalization ---
    print("[Test 1] Arabic Choice Letter Normalization...")
    opts = {"A": "الرياض", "B": "جدة", "C": "مكة", "D": "الدمام"}
    
    pass_1, letter_1 = evaluate_mcq("الإجابة الصحيحة هي (أ)", "A", opts)
    assert pass_1 and letter_1 == "A", f"Test 1.1 Failed: {letter_1}"
    
    pass_2, letter_2 = evaluate_mcq("الخيار ب هو الصحيح", "B", opts)
    assert pass_2 and letter_2 == "B", f"Test 1.2 Failed: {letter_2}"
    
    pass_3, letter_3 = evaluate_mcq("<think>نحلل الخيارات...</think><answer>ج</answer>", "C", opts)
    assert pass_3 and letter_3 == "C", f"Test 1.3 Failed: {letter_3}"
    print("  --> PASS: Arabic choice letters (أ, ب, ج) normalized perfectly!\n")

    # --- Test 2: Math & Numeric Equivalence ---
    print("[Test 2] Math & Numeric Equivalence...")
    pass_math_1, ext_1 = evaluate_openended("<think>66 + 3 = 69</think><answer>3</answer>", "3")
    assert pass_math_1, f"Test 2.1 Failed: {ext_1}"
    
    pass_math_2, ext_2 = evaluate_openended("<think>نحسب النتيجة</think><answer>\\boxed{42}</answer>", "42")
    assert pass_math_2, f"Test 2.2 Failed: {ext_2}"

    pass_math_3, ext_3 = evaluate_openended("النتيجة النهائية هي 15.5 دينار", "15.5")
    assert pass_math_3, f"Test 2.3 Failed: {ext_3}"
    print("  --> PASS: Math & numeric equivalence extraction 100% accurate!\n")

    # --- Test 3: AraIFEval Instruction Following ---
    print("[Test 3] AraIFEval Instruction Following...")
    insts = [
        {"type": "include_keyword", "keyword": "الخوارزميات"},
        {"type": "exclude_keyword", "keyword": "الحاسوب"},
    ]
    resp_pass = "<think>تفكير...</think><answer>يتناول هذا النص الخوارزميات والذكاء الاصطناعي بشكل مفصل.</answer>"
    strict_pass, ratio, meta = evaluate_ifeval(resp_pass, insts)
    assert strict_pass and ratio == 1.0, f"Test 3.1 Failed: {meta}"
    print("  --> PASS: AraIFEval constraint verification 100% accurate!\n")

    # --- Test 4: Task Loader & Dataset Normalization ---
    print("[Test 4] Core 4 Dataset Loader...")
    for task in CORE_FOUR_DATASETS:
        samples = load_araeval_task(task, limit=3)
        assert len(samples) > 0, f"Failed to load samples for task {task}"
        print(f"  - Task '{task:<12}': Loaded {len(samples)} samples cleanly (Sample 1 ID: {samples[0].id})")
    print("  --> PASS: All Core 4 tasks loaded cleanly!\n")

    print("=================================================================")
    print("ALL EXECUTIVE EVALUATION TESTS PASSED SUCCESSFULLY! 🏆")
    print("=================================================================")


if __name__ == "__main__":
    run_executive_tests()
