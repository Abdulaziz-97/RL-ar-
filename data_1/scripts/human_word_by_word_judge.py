import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
saved_dir = PACK_ROOT / "outputs" / "saved_benchmark_samples"

file_a = saved_dir / "model_a_deepseek_50.jsonl"
file_b = saved_dir / "model_b_qwen37_50.jsonl"

def load_items(path):
    if not path.exists():
        return {}
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items[item["id"]] = item
    return items

def analyze_sample(text):
    think = ""
    answer = ""
    if "<think>" in text and "</think>" in text:
        think = text.split("<think>")[1].split("</think>")[0].strip()
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[1].split("</answer>")[0].strip()
        
    words = think.split()
    word_count = len(words)
    has_equations = any(c in think for c in ["=", "×", "÷", "+", "-"])
    has_boilerplate = any(p in think for p in ["الخطوة التوضيحية", "يمكن حل هذه المسألة", "نراجع خطوات الحل"])
    
    return {
        "think": think,
        "answer": answer,
        "word_count": word_count,
        "has_equations": has_equations,
        "has_boilerplate": has_boilerplate
    }

def main():
    items_a = load_items(file_a)
    items_b = load_items(file_b)
    
    common_ids = sorted(list(set(items_a.keys()) & set(items_b.keys())))[:5]
    
    print("\n" + "="*90)
    print(" 👨‍⚖️ WORD-BY-WORD HUMAN JUDGE EVALUATION (SIDE-BY-SIDE EXAMPLES)")
    print("="*90)
    
    for idx, pid in enumerate(common_ids, 1):
        sample_a = items_a[pid]
        sample_b = items_b[pid]
        
        prompt = sample_a["prompt"]
        gold = sample_a["gold"]
        
        an_a = analyze_sample(sample_a["response"])
        an_b = analyze_sample(sample_b["response"])
        
        print(f"\n--- [PROMPT #{idx} | ID: {pid} | Domain: {sample_a.get('domain', 'gsm8k')}] ---")
        print(f"📌 QUESTION: {prompt}")
        print(f"🎯 GOLD ANSWER: {gold}")
        
        print("\n🔹 MODEL A (DeepSeek Pro):")
        print(f"  [Word Count: {an_a['word_count']} words | Answer: '{an_a['answer']}']")
        print(f"  THINKING SNIPPET:\n  \"{an_a['think'][:280]}...\"")
        
        print("\n🔹 MODEL B (Qwen 3.7 Flash):")
        print(f"  [Word Count: {an_b['word_count']} words | Answer: '{an_b['answer']}']")
        print(f"  THINKING SNIPPET:\n  \"{an_b['think'][:280]}...\"")
        
        print("-" * 90)

if __name__ == "__main__":
    main()
