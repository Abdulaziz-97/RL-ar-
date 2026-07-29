#!/usr/bin/env python3
"""
Filter training dataset to create `data/arabic_reasoning_rlvr_hard_v3.jsonl`.

Removes trivial 1-2 step problems where base_pass@8 = 1.0 (zero reward variance),
and tags remaining problems with 'medium' and 'hard' difficulty tags for Curriculum Learning.
"""

import json
from pathlib import Path

INPUT_PATH = Path("data/arabic_reasoning_rlvr_train.jsonl")
OUTPUT_PATH = Path("data/arabic_reasoning_rlvr_hard_v3.jsonl")

def derive_difficulty(record: dict) -> tuple[str, bool]:
    """
    Returns (difficulty_tag, is_keep).
    
    Eliminates trivial 1-2 step problems (which have pass@8 = 1.0, 0 reward variance).
    Categorizes remaining problems into 'medium' or 'hard' for Curriculum Learning.
    """
    metadata = record.get("metadata", {})
    num_steps = metadata.get("num_steps", 3)
    grade = str(metadata.get("grade_level", "")).lower()
    
    # Check if explicit pass@8 evaluation exists in metadata
    pass_8 = metadata.get("base_pass_8") or metadata.get("pass_8")
    if pass_8 is not None and float(pass_8) >= 0.875:
        return ("trivial", False) # Drop zero-variance trivial items
        
    # Heuristic filtering based on step complexity & grade level
    if isinstance(num_steps, (int, float)):
        if num_steps <= 2 and ("grade_1" in grade or "grade_2" in grade or "grade_3" in grade or "grade_4" in grade):
            return ("trivial", False) # Exclude trivial grade 1-4 short step questions
        elif num_steps <= 4:
            return ("medium", True)
        else:
            return ("hard", True)
            
    return ("medium", True)

def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    records = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    kept_records = []
    counts = {"trivial": 0, "medium": 0, "hard": 0}

    for rec in records:
        tag, keep = derive_difficulty(rec)
        counts[tag] = counts.get(tag, 0) + 1
        if keep:
            if "metadata" not in rec:
                rec["metadata"] = {}
            rec["metadata"]["difficulty_tag"] = tag
            rec["difficulty_tag"] = tag
            kept_records.append(rec)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Dataset Curation Summary:")
    print(f"  Total Input Prompts: {len(records)}")
    print(f"  Filtered Out Trivial (pass@8=1.0): {counts.get('trivial', 0)}")
    print(f"  Kept Medium Prompts: {counts.get('medium', 0)}")
    print(f"  Kept Hard Prompts: {counts.get('hard', 0)}")
    print(f"  Total Retained Hard V3 Prompts: {len(kept_records)}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
