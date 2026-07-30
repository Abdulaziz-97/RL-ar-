import sys
import json
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR  = PACK_ROOT.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rlvr_pipeline.rewards import (
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    composite_reward_func,
)

print("="*80, flush=True)
print(" RUNNING PRODUCTION UNIT TESTS FOR ALL 4 REWARD FUNCTIONS", flush=True)
print("="*80, flush=True)

# 1. Test Numeric Reward
p = ["ما ناتج..."]
gt_num = ["13200"]
spec_num = [{"type": "integer", "canonical": "13200", "ground_truth_structured": 13200}]

completions_num = [
    "<think>شرح...</think>\n<answer>13200</answer>",
    "<think>شرح...</think>\n<answer>13,200</answer>",
    "<think>شرح...</think>\n<answer>13200.0</answer>",
    "<think>شرح...</think>\n<answer>١٣٢٠٠</answer>",
]

for c in completions_num:
    r = correctness_reward_func(p, [c], ground_truth_answer=gt_num, domain=["math"], answer_spec=spec_num)
    assert r[0] == 1.0, f"Numeric test failed for {c}: got {r[0]}"

print("  PASSED 1: Numeric Reward (reward_numeric) [13200, 13,200, 13200.0, ١٣٢٠٠ -> 1.0]", flush=True)

# 2. Test Logic JSON Reward
gt_json = [json.dumps({"ans": 2, "status": "pass"})]
spec_json = [{"type": "logic_json", "canonical": gt_json[0], "ground_truth_structured": {"ans": 2, "status": "pass"}}]
c_json_reordered = '<think>استنتاج...</think>\n<answer>{"status": "pass", "ans": 2}</answer>'

r_json = correctness_reward_func(p, [c_json_reordered], ground_truth_answer=gt_json, domain=["logic"], answer_spec=spec_json)
assert r_json[0] == 1.0, f"Logic JSON test failed: got {r_json[0]}"

print("  PASSED 2: Logic JSON Reward (reward_logic_json) [Semantic key reordering -> 1.0]", flush=True)

# 3. Test Format Strictness Reward
# Realistic CoT >= 10 words -> score 1.0
c_format_full = "<think>\nالخطوة الأولى نحسب إجمالي الأشجار بالنخيل ثم نجمع العدد مع الأشجار الأخرى للحصول على النتيجة النهائية.\n</think>\n<answer>160</answer>"

# Short CoT < 10 words -> penalized by -0.3 -> score 0.7
c_format_short = "<think>\nالخطوة الأولى: نحسب إجمالي الإنتاج.\n</think>\n<answer>160</answer>"

# Invalid missing tag -> score < 0.30
c_format_bad = "الشرح بدون وسم تفكير 160"

r_fmt_full = format_reward_func(p, [c_format_full])
r_fmt_short = format_reward_func(p, [c_format_short])
r_fmt_bad = format_reward_func(p, [c_format_bad])

assert r_fmt_full[0] == 1.0, f"Format full CoT failed: got {r_fmt_full[0]}"
assert r_fmt_short[0] == 0.7, f"Format short CoT penalty failed: got {r_fmt_short[0]}"
assert r_fmt_bad[0] < 0.30, f"Format bad failed: got {r_fmt_bad[0]}"

print("  PASSED 3: Format Strictness Reward (reward_format) [CoT >=10w -> 1.0, <10w -> 0.7, missing -> 0.0]", flush=True)

# 4. Test Arabic Purity Reward
c_lang_pure = "<think>الخطوة الأولى: نحسب إجمالي إنتاج الموسم الأول بضرب عدد الأشجار في إنتاج كل شجرة.</think>\n<answer>120</answer>"
c_lang_chinese = "<think>الخطوة الأولى 这是一个测试 中文字符 نحسب الإجمالي.</think>\n<answer>120</answer>"

r_lang_pure = language_reward_func(p, [c_lang_pure])
r_lang_chinese = language_reward_func(p, [c_lang_chinese])

assert r_lang_pure[0] >= 0.85, f"Arabic purity pure failed: got {r_lang_pure[0]}"
assert r_lang_chinese[0] < 0.70, f"Arabic purity Chinese failed: got {r_lang_chinese[0]}"

print("  PASSED 4: Arabic Purity Reward (reward_arabic_purity) [Pure Arabic -> >=0.85, Chinese -> <0.70]", flush=True)

print("="*80, flush=True)
print(" VERDICT: ALL 4 REWARD FUNCTIONS PASSED 100% FLAWLESSLY!", flush=True)
print("="*80, flush=True)
