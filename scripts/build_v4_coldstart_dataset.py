import json
from pathlib import Path

def main():
    base_coldstart_path = Path("data/arabic_reasoning_coldstart_train.jsonl")
    hard_math_path = Path("data/arabic_reasoning_coldstart_hard_math_v4.jsonl")
    output_path = Path("data/arabic_reasoning_coldstart_v4.jsonl")

    samples = []
    seen_prompts = set()

    # 1. Read base 1,707 cold-start samples
    if base_coldstart_path.exists():
        with open(base_coldstart_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt = item.get("prompt") or item.get("question") or ""
                    if prompt not in seen_prompts:
                        seen_prompts.add(prompt)
                        samples.append(item)
        print(f"[INFO] Loaded {len(samples)} base cold-start samples from {base_coldstart_path}")

    # 2. Append new 500 hard reasoning samples
    added_count = 0
    if hard_math_path.exists():
        with open(hard_math_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt = item.get("prompt") or item.get("question") or ""
                    if prompt not in seen_prompts:
                        seen_prompts.add(prompt)
                        samples.append(item)
                        added_count += 1
        print(f"[INFO] Merged {added_count} new hard reasoning samples from {hard_math_path}")

    # 3. Save master 2,207 dataset
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved combined V4 Master Cold-Start Dataset ({len(samples)} samples) to: {output_path}")

if __name__ == "__main__":
    main()
