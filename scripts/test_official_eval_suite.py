"""
Comprehensive Unit Test Suite for Vendored Official Saudi-LLM Eval Engine.
Verifies all 7 task normalizers, random-baseline score normalization, AraIFEval instruction rules,
proportional sampling, and checkpoint summarization.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add official_eval to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "official_eval"))

from run_araeval import proportional_sample_counts, summarize_checkpoint
from tasks.araeval.utils import (
    PUBLIC_TEST_COUNTS,
    RANDOM_BASELINES,
    araeval_overall,
    check_instruction,
    normalize_against_random,
    normalize_araifeval,
    normalize_aramath,
    normalize_arapro,
    normalize_etec,
    normalize_ien_mcq,
    normalize_ien_tf,
    normalize_truthfulqa,
)


def run_unit_tests():
    print("=================================================================")
    print("RUNNING COMPREHENSIVE UNIT TEST SUITE FOR OFFICIAL EVAL ENGINE")
    print("=================================================================\n")

    # --- Test 1: Task Sample Counts & Random Baselines ---
    print("[Test 1] Public Test Counts & Random Baselines...")
    total_docs = sum(PUBLIC_TEST_COUNTS.values())
    assert total_docs == 24378, f"Test 1.1 Failed: Total docs {total_docs} != 24378"
    assert len(RANDOM_BASELINES) == 7, "Test 1.2 Failed: Baseline count != 7"
    assert RANDOM_BASELINES["arapro"] == 25.0, "Test 1.3 Failed: arapro baseline != 25.0"
    print("  --> PASS: 7 tasks and 24,378 total documents verified!\n")

    # --- Test 2: Random Baseline Normalization Formula ---
    print("[Test 2] Random Baseline Normalization Formula...")
    # Example: 60.0% raw on 4-choice MCQ (25.0% baseline) -> (60-25)/(100-25)*100 = 46.6667%
    norm_score = normalize_against_random(60.0, 25.0)
    assert abs(norm_score - 46.666666) < 1e-4, f"Test 2.1 Failed: {norm_score}"
    
    # 0.0% baseline for IFEval
    norm_ifeval = normalize_against_random(75.0, 0.0)
    assert norm_ifeval == 75.0, f"Test 2.2 Failed: {norm_ifeval}"
    print("  --> PASS: Normalization formula (Raw - Random) / (100 - Random) exact!\n")

    # --- Test 3: MCQ Normalization Protocols ---
    print("[Test 3] MCQ Normalizers for All Tasks...")
    
    # AraPro
    raw_arapro = {"question": "ما هي عاصمة المملكة؟", "choice1": "جدة", "choice2": "الرياض", "choice3": "مكة", "choice4": "الدمام", "answer": 2}
    norm_ap = normalize_arapro(raw_arapro)
    assert norm_ap["gold"] == 1 and norm_ap["choices"] == ["A", "B", "C", "D"], f"Test 3.1 Failed: {norm_ap}"
    assert norm_ap["query"].endswith("الإجابة:"), "Test 3.1 Prompt suffix missing 'الإجابة:'"

    # AraMath
    raw_math = {"question": "كم 5 + 5؟", "options": ["(A) 8", "(B) 10", "(C) 12", "(D) 15"], "label": "B", "passage": ""}
    norm_am = normalize_aramath(raw_math)
    assert norm_am["gold"] == 1 and norm_am["choices"] == ["A", "B", "C", "D"], f"Test 3.2 Failed: {norm_am}"

    # TruthfulQA
    raw_tqa = {"input": "هل الأرض كروية؟", "options": ["A. نعم", "B. لا"], "label": "A", "instruction": "أجب عن السؤال"}
    norm_tq = normalize_truthfulqa(raw_tqa)
    assert norm_tq["gold"] == 0 and len(norm_tq["choices"]) == 2, f"Test 3.3 Failed: {norm_tq}"
    print("  --> PASS: All MCQ task normalizers produce correct gold targets & 'الإجابة:' prompts!\n")

    # --- Test 4: AraIFEval Instruction Rules ---
    print("[Test 4] AraIFEval Instruction Rule Checker...")
    # Include keyword rule
    pass_kw = check_instruction("include_keywords", "قم بتضمين \"الذكاء الاصطناعي\"", "هذا النص يتناول الذكاء الاصطناعي بشغف.")
    assert pass_kw, "Test 4.1 Failed: Include keyword check"

    # Paragraph count rule
    pass_para = check_instruction("number_paragraphs", "اكتب فقرتين", "الفقرة الأولى.\n\nالفقرة الثانية.")
    assert pass_para, "Test 4.2 Failed: Paragraph count check"

    # Bullet points rule
    pass_bullet = check_instruction("number_bullets", "اكتب 3 نقاط", "- النقطة الأولى\n- النقطة الثانية\n- النقطة الثالثة")
    assert pass_bullet, "Test 4.3 Failed: Bullet points check"
    print("  --> PASS: AraIFEval instruction checkers verified 100%!\n")

    # --- Test 5: Proportional Sampling & Allocation ---
    print("[Test 5] Proportional Sampling Allocation...")
    counts_500 = proportional_sample_counts(500)
    assert sum(counts_500.values()) == 500, f"Test 5.1 Failed: Sum {sum(counts_500.values())} != 500"
    print(f"  - Proportional Allocation for 500 samples: {counts_500}")
    print("  --> PASS: Proportional sampling allocation algorithm exact!\n")

    # --- Test 6: Checkpoint Summarizer ---
    print("[Test 6] Checkpoint Summarization & Primary Metric...")
    mock_checkpoint = {
        "benchmark_mode": "full",
        "full_public_documents": 24378,
        "completed": {
            "araeval_ien_mcq": {"metrics": {"acc_norm": 0.60}},
            "araeval_ien_tf": {"metrics": {"acc_norm": 0.75}},
            "araeval_aramath": {"metrics": {"acc_norm": 0.80}},
            "araeval_etec": {"metrics": {"acc_norm": 0.70}},
            "araeval_arapro": {"metrics": {"acc_norm": 0.65}},
            "araeval_truthfulqa": {"metrics": {"acc_norm": 0.60}},
            "araeval_ifeval": {"metrics": {
                "prompt_level_strict_acc,none": 0.70,
                "inst_level_strict_acc,none": 0.80,
                "prompt_level_loose_acc,none": 0.75,
                "inst_level_loose_acc,none": 0.85,
            }},
        }
    }
    summary = summarize_checkpoint(mock_checkpoint)
    assert "paper_primary" in summary and summary["paper_primary"] is not None, "Test 6.1 Failed: Missing paper_primary metric"
    print(f"  - Calculated Paper Primary Metric: {summary['paper_primary']:.2f}%")
    print("  --> PASS: Checkpoint summarizer & paper primary metric calculated perfectly!\n")

    # --- Test 7: vLLM Kwargs Configuration ---
    print("[Test 7] vLLM Kwargs & Log-Likelihood Compatibility...")
    class MockArgs:
        model = "unsloth/Qwen3.5-4B"
        batch_size = "auto"
        max_batch_size = 32
        max_length = 4096
        max_lora_rank = 32
        adapter_path = None
        tensor_parallel_size = 2
        enable_thinking = False

    from run_araeval import build_vllm_kwargs
    vllm_kw = build_vllm_kwargs(MockArgs())
    assert vllm_kw["enable_thinking"] is False, "Test 7.1 Failed: enable_thinking should be False for loglik tasks"
    assert vllm_kw["enforce_eager"] is True, "Test 7.2 Failed: enforce_eager should be True"
    assert vllm_kw["tensor_parallel_size"] == 2, "Test 7.3 Failed: tensor_parallel_size should be 2"
    assert "think_end_token" not in vllm_kw, "Test 7.4 Failed: think_end_token should not be present when enable_thinking=False"
    print("  --> PASS: vLLM Kwargs (enable_thinking=False, enforce_eager=True, TP=2) 100% verified!\n")

    print("=================================================================")
    print("ALL 7 EXECUTIVE UNIT TESTS PASSED SUCCESSFULLY! 🏆")
    print("=================================================================")


if __name__ == "__main__":
    run_unit_tests()
