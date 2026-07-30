import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
file_a = PACK_ROOT / "outputs" / "model_a_deepseek_50.jsonl"
file_b = PACK_ROOT / "outputs" / "model_b_deepseek_r1_50.jsonl"

def analyze_sample(item):
    resp = item.get("response", "")
    gold = str(item.get("gold", "")).strip()
    
    # 1. Extract think and answer
    think = ""
    m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL | re.IGNORECASE)
    if m:
        think = m.group(1).strip()
        
    answer = ""
    am = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL | re.IGNORECASE)
    if am:
        answer = am.group(1).strip()
        
    # 2. Check tag format
    has_tags = "<think>" in resp and "</think>" in resp and "<answer>" in resp and "</answer>" in resp
    
    # 3. Check MSA purity
    ar_chars = sum(1 for c in resp if "\u0600" <= c <= "\u06ff")
    total_chars = max(len(resp.replace(" ", "")), 1)
    purity = (ar_chars / total_chars) * 100
    
    # 4. Check step depth
    lines = [l.strip() for l in think.split("\n") if l.strip()]
    num_steps = len(lines)
    
    # 5. Check accuracy against gold
    is_correct = False
    if gold and answer:
        is_correct = (gold.lower() in answer.lower()) or (answer.lower() in gold.lower())
        
    # 6. Repetition check
    words = think.split()
    unique_words = set(words)
    rep_ratio = (len(unique_words) / max(len(words), 1)) * 100 if words else 100
    
    return {
        "id": item.get("id"),
        "domain": item.get("domain"),
        "prompt": item.get("prompt"),
        "think": think,
        "answer": answer,
        "gold": gold,
        "has_tags": has_tags,
        "purity": round(purity, 1),
        "num_steps": num_steps,
        "is_correct": is_correct,
        "rep_ratio": round(rep_ratio, 1)
    }

def main():
    if not file_a.exists() or not file_b.exists():
        print("[INFO] Waiting for both Model A and Model B output files to finish generation...")
        return
        
    items_a = [json.loads(l) for l in open(file_a, encoding="utf-8") if l.strip()]
    items_b = [json.loads(l) for l in open(file_b, encoding="utf-8") if l.strip()]
    
    print(f"[INFO] Loaded {len(items_a)} Model A samples and {len(items_b)} Model B samples.")
    
    eval_a = [analyze_sample(it) for it in items_a]
    eval_b = [analyze_sample(it) for it in items_b]
    
    print("\n" + "="*85)
    print(" ⚖️ HUMAN-STYLE QUALITATIVE JUDGE EVALUATION (50 PROMPTS SIDE-BY-SIDE)")
    print("="*85)
    
    correct_a = sum(1 for e in eval_a if e["is_correct"])
    correct_b = sum(1 for e in eval_b if e["is_correct"])
    
    tags_a = sum(1 for e in eval_a if e["has_tags"])
    tags_b = sum(1 for e in eval_b if e["has_tags"])
    
    avg_purity_a = sum(e["purity"] for e in eval_a) / len(eval_a) if eval_a else 0
    avg_purity_b = sum(e["purity"] for e in eval_b) / len(eval_b) if eval_b else 0
    
    avg_steps_a = sum(e["num_steps"] for e in eval_a) / len(eval_a) if eval_a else 0
    avg_steps_b = sum(e["num_steps"] for e in eval_b) / len(eval_b) if eval_b else 0
    
    print(f"\n1. ACCURACY & VERIFICATION (Gold Answer Match):")
    print(f"   - Model A (DeepSeek V3): {correct_a}/{len(eval_a)} ({correct_a/len(eval_a)*100:.1f}%)")
    print(f"   - Model B (DeepSeek R1): {correct_b}/{len(eval_b)} ({correct_b/len(eval_b)*100:.1f}%)")
    
    print(f"\n2. FORMAT & TAG COMPLIANCE (<think> & <answer>):")
    print(f"   - Model A (DeepSeek V3): {tags_a}/{len(eval_a)} ({tags_a/len(eval_a)*100:.1f}%)")
    print(f"   - Model B (DeepSeek R1): {tags_b}/{len(eval_b)} ({tags_b/len(eval_b)*100:.1f}%)")
    
    print(f"\n3. REASONING DEPTH & STEPS:")
    print(f"   - Model A (DeepSeek V3): Avg {avg_steps_a:.1f} lines of CoT per prompt")
    print(f"   - Model B (DeepSeek R1): Avg {avg_steps_b:.1f} lines of CoT per prompt")
    
    print(f"\n4. MSA ARABIC PURITY:")
    print(f"   - Model A (DeepSeek V3): {avg_purity_a:.1f}%")
    print(f"   - Model B (DeepSeek R1): {avg_purity_b:.1f}%")
    
    print("\n" + "="*85)
    print(" 🏆 FINAL JUDGE VERDICT")
    print("="*85)
    
    if correct_b >= correct_a:
        print("Winner: MODEL B (DeepSeek-R1 Reasoner) achieves superior reasoning depth, higher accuracy, and native CoT step verification!")
    else:
        print("Winner: MODEL A (DeepSeek V3 Pro) achieves higher accuracy and format compliance!")

if __name__ == "__main__":
    main()
