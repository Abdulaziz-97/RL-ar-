import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from vendor.synth.dspy_teacher import think_quality_ok, parse_gt
from vendor.answer_match import match_answers

file_a = PACK_ROOT / "outputs" / "model_a_deepseek_50.jsonl"

def audit_sample(item):
    resp = item.get("response", "")
    gt_obj = item.get("gold")
    domain = item.get("domain", "gsm8k")
    
    # 1. Format Check
    has_think = "<think>" in resp and "</think>" in resp
    has_answer = "<answer>" in resp and "</answer>" in resp
    format_ok = has_think and has_answer
    
    think_text = ""
    ans_text = ""
    if has_think:
        think_text = resp.split("<think>")[1].split("</think>")[0].strip()
    if has_answer:
        ans_text = resp.split("<answer>")[1].split("</answer>")[0].strip()
        
    # 2. Purity & Quality Audit (dspy_teacher rules)
    quality_pass, quality_reason = think_quality_ok(think_text, resp, domain=domain, gt=gt_obj)
    
    # 3. Answer Match
    answer_pass = False
    if ans_text and gt_obj is not None:
        answer_pass = match_answers(ans_text, str(gt_obj), domain=domain)
        
    # 4. Leak Check
    no_leak = str(gt_obj) not in think_text if domain != "logic" else True
    
    passed_all = format_ok and quality_pass and answer_pass and no_leak
    
    return {
        "id": item.get("id"),
        "format_ok": format_ok,
        "quality_pass": quality_pass,
        "quality_reason": quality_reason,
        "answer_pass": answer_pass,
        "no_leak": no_leak,
        "passed_all": passed_all
    }

def main():
    if not file_a.exists():
        print(f"[ERROR] File not found: {file_a}")
        return
        
    items = [json.loads(l) for l in open(file_a, encoding="utf-8") if l.strip()]
    audits = [audit_sample(it) for it in items]
    
    total = len(audits)
    format_passed = sum(1 for a in audits if a["format_ok"])
    quality_passed = sum(1 for a in audits if a["quality_pass"])
    answer_passed = sum(1 for a in audits if a["answer_pass"])
    no_leak_passed = sum(1 for a in audits if a["no_leak"])
    all_passed = sum(1 for a in audits if a["passed_all"])
    
    print("\n" + "="*80)
    print(f" 🛡️ MODEL A (DeepSeek Pro) RELEASE GATES AUDIT REPORT ({total} SAMPLES)")
    print("="*80)
    print(f"  1. Format Gate (<think> & <answer>):      {format_passed}/{total} ({format_passed/total*100:.1f}%)")
    print(f"  2. Quality & MSA Purity Gate:             {quality_passed}/{total} ({quality_passed/total*100:.1f}%)")
    print(f"  3. Ground-Truth Match Gate:               {answer_passed}/{total} ({answer_passed/total*100:.1f}%)")
    print(f"  4. Zero Answer Leak Gate:                 {no_leak_passed}/{total} ({no_leak_passed/total*100:.1f}%)")
    print("-" * 80)
    print(f"  🏆 PASSED ALL 5 RELEASE GATES:            {all_passed}/{total} ({all_passed/total*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
