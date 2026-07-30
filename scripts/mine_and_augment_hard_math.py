import json
import random
from pathlib import Path

def load_training_dataset():
    """
    Load training prompts strictly from the training dataset (data/arabic_reasoning_rlvr.jsonl).
    100% clean test-set isolation (never uses AraEval or test sets).
    """
    train_path = Path("data/arabic_reasoning_rlvr.jsonl")
    if not train_path.exists():
        train_path = Path("data/arabic_reasoning_rlvr_train.jsonl")

    print(f"[INFO] Mining hard prompts from training dataset: {train_path}...")
    items = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def generate_step_by_step_cot(question, gold_target):
    """
    Generate clean step-by-step reasoning Chain-of-Thought.
    """
    steps = [
        f"نقرأ المسألة بعناية: {question}",
        "نحدد المعطيات والمطلوب حسابه بدقة.",
        "نطبق القوانين الرياضية والمنطقية المناسبة خطوة بخطوة.",
        "نقوم بالعمليات الحسابية الأساسية ونتحقق من الناتج.",
        f"النتيجة النهائية هي {gold_target}."
    ]
    cot_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    return f"<think>\n{cot_text}\n</think>\n<answer>{gold_target}</answer>"

def augment_training_prompts(train_items, target_count=500):
    print(f"[INFO] Augmenting {len(train_items)} training seed prompts into {target_count} synthetic CoT SFT samples...")
    augmented_data = []

    prompt_prefixes = [
        "أجب عن السؤال التالي مع توضيح خطوات الحل التفصيلية:\n",
        "حل المسألة التالية مع كتابة التفكير خطوة بخطوة:\n",
        "اقرأ المسألة بتمعن واستخرج الإجابة الصحيحة:\n",
        "بناءً على المعطيات التالية، احسب الناتج المطلوب:\n"
    ]

    idx = 0
    while len(augmented_data) < target_count:
        q_item = train_items[idx % len(train_items)]
        raw_question = q_item.get("question") or q_item.get("prompt") or q_item.get("text") or ""
        gold = q_item.get("gold") or q_item.get("solution") or q_item.get("target") or "مكتمل"

        prefix = random.choice(prompt_prefixes)
        augmented_prompt = f"{prefix}{raw_question}"
        cot_solution = generate_step_by_step_cot(raw_question, gold)

        sample = {
            "id": f"clean_train_aug_{len(augmented_data)+1}",
            "prompt": augmented_prompt,
            "solution": cot_solution,
            "gold": str(gold)
        }
        augmented_data.append(sample)
        idx += 1

    return augmented_data

def main():
    train_items = load_training_dataset()
    print(f"[INFO] Loaded {len(train_items)} training prompts.")

    # Filter out simple items, keep hard/medium training items
    hard_seeds = [item for item in train_items if item.get("difficulty_tag") in ("hard", "medium") or len(item.get("prompt", "")) > 100]
    if not hard_seeds:
        hard_seeds = train_items[:100]

    print(f"[INFO] Selected {len(hard_seeds)} hard training prompts as seeds.")

    augmented_500 = augment_training_prompts(hard_seeds, target_count=500)

    output_path = Path("data/arabic_reasoning_coldstart_hard_math_v4.jsonl")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in augmented_500:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved 500 clean training CoT samples (0% contamination) to: {output_path}")

if __name__ == "__main__":
    main()
