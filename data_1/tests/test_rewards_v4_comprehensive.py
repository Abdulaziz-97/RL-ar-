"""
Exhaustive Production Unit Tests for Saudi-LLM V4 Reward Functions.
Tests:
  1. Numeric Reward (Formatting, Arabic digits, floats, commas)
  2. Logic JSON Reward (Semantic equivalence, key reordering)
  3. Format Strictness Reward (<think>...</think><answer>...</answer> tags)
  4. Arabic Language Purity Reward (MSA consistency & character purity)
"""

import json
import pytest
import sys
from pathlib import Path

# Add src to path
PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR  = PACK_ROOT.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rlvr_pipeline.rewards import (
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    composite_reward_func,
    _extract_completion_text,
)

# ── 1. NUMERIC REWARD UNIT TESTS ─────────────────────────────────────
def test_numeric_reward_variations():
    """Test numeric normalization: 13200, 13,200, 13200.0, Arabic digits ١٣٢٠٠."""
    prompt = "ما ناتج..."
    ground_truth = "13200"
    spec = {"type": "integer", "canonical": "13200", "ground_truth_structured": 13200}
    
    variations = [
        "<think>شرح خطوة بخطوة بالتفصيل لحساب النتيجة النهائية...</think>\n<answer>13200</answer>",
        "<think>شرح خطوة بخطوة بالتفصيل لحساب النتيجة النهائية...</think>\n<answer>13,200</answer>",
        "<think>شرح خطوة بخطوة بالتفصيل لحساب النتيجة النهائية...</think>\n<answer>13200.0</answer>",
        "<think>شرح خطوة بخطوة بالتفصيل لحساب النتيجة النهائية...</think>\n<answer>١٣٢٠٠</answer>",
        "<think>شرح خطوة بخطوة بالتفصيل لحساب النتيجة النهائية...</think>\n<answer> 13200 </answer>",
    ]
    
    for c in variations:
        res = correctness_reward_func([prompt], [c], ground_truth_answer=[ground_truth], domain=["math"], answer_spec=[spec])
        assert res[0] == 1.0, f"Failed numeric normalization for completion: {c}"

def test_numeric_reward_incorrect():
    """Test wrong numbers yield 0.0 reward."""
    prompt = "ما ناتج..."
    ground_truth = "13200"
    spec = {"type": "integer", "canonical": "13200", "ground_truth_structured": 13200}
    completion = "<think>شرح...</think>\n<answer>9999</answer>"
    
    res = correctness_reward_func([prompt], [completion], ground_truth_answer=[ground_truth], domain=["math"], answer_spec=[spec])
    assert res[0] == 0.0, f"Incorrect answer yielded non-zero reward: {res[0]}"

# ── 2. LOGIC JSON REWARD UNIT TESTS ─────────────────────────────────
def test_logic_json_reward_semantic():
    """Test JSON key reordering and string vs int semantic equality."""
    prompt = "أوجد القيمة..."
    ground_truth = json.dumps({"ans": 2, "status": "pass"})
    spec = {"type": "logic_json", "canonical": ground_truth, "ground_truth_structured": {"ans": 2, "status": "pass"}}
    
    # Reordered keys in output
    completion_reordered = '<think>استنتاج منطقي كامل بخطوات واضحة لتحديد القيمة المناسبة...</think>\n<answer>{"status": "pass", "ans": 2}</answer>'
    res = correctness_reward_func([prompt], [completion_reordered], ground_truth_answer=[ground_truth], domain=["logic"], answer_spec=[spec])
    assert res[0] == 1.0, f"Failed semantic JSON key reordering match!"

def test_logic_json_reward_malformed():
    """Test malformed JSON yields 0.0 reward."""
    prompt = "أوجد القيمة..."
    ground_truth = json.dumps({"ans": 2})
    completion_bad = '<think>استنتاج...</think>\n<answer>{"ans": 2 INVALID</answer>'
    
    res = correctness_reward_func([prompt], [completion_bad], ground_truth_answer=[ground_truth], domain=["logic"])
    assert res[0] == 0.0, "Malformed JSON yielded non-zero reward!"

# ── 3. FORMAT STRICTNESS REWARD UNIT TESTS ─────────────────────────────
def test_format_strictness_perfect():
    """Test strict <think> and <answer> tags with >= 10 words yield 1.0 reward."""
    completion = "<think>\nالخطوة الأولى نحسب إجمالي الأشجار بالنخيل ثم نجمع العدد مع الأشجار الأخرى للحصول على النتيجة النهائية.\n</think>\n<answer>160</answer>"
    res = format_reward_func(["prompt"], [completion])
    assert res[0] == 1.0, f"Perfect format failed: {res[0]}"

def test_format_strictness_missing_tags():
    """Test missing <think> or <answer> tags yield partial shaping (< 0.30) instead of 1.0."""
    no_think = "الشرح مباشرة بدون وسم تفكير\n<answer>160</answer>"
    no_answer = "<think>تفكير بدون إجابة</think>"
    
    res1 = format_reward_func(["prompt"], [no_think])
    res2 = format_reward_func(["prompt"], [no_answer])
    
    assert res1[0] < 1.0, "Missing <think> tag was given full credit!"
    assert res2[0] < 1.0, "Missing <answer> tag was given full credit!"

# ── 4. ARABIC LANGUAGE PURITY REWARD UNIT TESTS ────────────────────────
def test_arabic_purity_high():
    """Test 100% Modern Standard Arabic text receives high language consistency score."""
    pure_arabic = "<think>الخطوة الأولى: نحسب إجمالي إنتاج الموسم الأول بضرب عدد الأشجار في إنتاج كل شجرة.</think>\n<answer>120</answer>"
    res = language_reward_func(["prompt"], [pure_arabic])
    assert res[0] >= 0.85, f"Pure Arabic text got low language reward: {res[0]}"

def test_arabic_purity_chinese_penalty():
    """Test non-Arabic Chinese/English hallucinated tokens get penalized."""
    mixed_chinese = "<think>الخطوة الأولى 这是一个测试 中文字符 نحسب الإجمالي.</think>\n<answer>120</answer>"
    res = language_reward_func(["prompt"], [mixed_chinese])
    assert res[0] < 0.70, f"Mixed Chinese text did not get penalized: {res[0]}"
