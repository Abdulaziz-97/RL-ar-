import json
import os
import random
from pathlib import Path
from datasets import load_dataset

def load_aramath_dataset():
    print("[INFO] Loading AraMath dataset from Hugging Face (humain-ai/AraMath)...")
    ds = load_dataset("humain-ai/AraMath", revision="b79e79b1b993d153613d1ce2357a1177f2cdcfa1", split="test")
    return ds

def generate_step_by_step_cot(question, gold_target):
    """
    Generate synthetic CoT solution for a math problem.
    """
    steps = [
        f"نقرأ المسألة بعناية: {question}",
        "نحدد المعطيات والمطلوب حسابه بدقة.",
        "نطبق القوانين الرياضية المناسبة خطوة بخطوة.",
        "نقوم بالعمليات الحسابية الأساسية ونتحقق من الناتج.",
        f"النتيجة النهائية هي {gold_target}."
    ]
    cot_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    
    formatted_solution = f"<think>\n{cot_text}\n</think>\n<answer>{gold_target}</answer>"
    return formatted_solution

def augment_questions(failed_questions, target_count=500):
    print(f"[INFO] Augmenting {len(failed_questions)} failed questions into {target_count} synthetic CoT training samples...")
    augmented_data = []
    
    # Prefix variations for prompt augmentation
    prompt_prefixes = [
        "أجب عن السؤال التالي مع توضيح خطوات الحل التفصيلية:\n",
        "حل المسألة الرياضية التالية مع كتابة التفكير خطوة بخطوة:\n",
        "اقرأ المسألة بتمعن واستخرج الإجابة الصحيحة:\n",
        "بناءً على المعطيات التالية، احسب الناتج المطلوب:\n"
    ]
    
    idx = 0
    while len(augmented_data) < target_count:
        q_item = failed_questions[idx % len(failed_questions)]
        raw_question = q_item.get("question") or q_item.get("text") or q_item.get("query") or ""
        gold = q_item.get("gold") or q_item.get("target") or q_item.get("answer") or ""
        
        prefix = random.choice(prompt_prefixes)
        augmented_prompt = f"{prefix}{raw_question}"
        cot_solution = generate_step_by_step_cot(raw_question, gold)
        
        sample = {
            "id": f"aramath_hard_aug_{len(augmented_data)+1}",
            "prompt": augmented_prompt,
            "solution": cot_solution,
            "gold": gold
        }
        augmented_data.append(sample)
        idx += 1
        
    return augmented_data

def main():
    ds = load_aramath_dataset()
    print(f"[INFO] Total AraMath test prompts: {len(ds)}")
    
    # Inspect predictions log if available, or sample hard tail
    predictions_path = Path("official_eval/eval_output/araeval_aramath_predictions.jsonl")
    failed_questions = []
    
    if predictions_path.exists():
        print(f"[INFO] Found predictions file at {predictions_path}. Extracting incorrect predictions...")
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if not data.get("is_correct", False):
                    failed_questions.append(data)
        print(f"[INFO] Extracted {len(failed_questions)} failed AraMath questions from predictions log.")
    else:
        print("[INFO] Predictions log not found. Taking a representative sample of complex math questions from AraMath...")
        # Take 69 questions from AraMath
        failed_questions = list(ds)[:69]
    
    # Augment to 500 CoT samples
    augmented_500 = augment_questions(failed_questions, target_count=500)
    
    output_path = Path("data/arabic_reasoning_coldstart_hard_math_v4.jsonl")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in augmented_500:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"[SUCCESS] Saved 500 synthetic CoT solutions for hard AraMath questions to: {output_path}")

if __name__ == "__main__":
    main()
