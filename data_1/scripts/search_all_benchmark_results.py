import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

roots = [
    Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B"),
    Path(r"C:\Users\Azooo\RL-ar--1")
]

print("="*80)
print(" SEARCHING FOR ALL BENCHMARK & EVALUATION RESULTS ACROSS DRIVES")
print("="*80)

found_files = []
for root in roots:
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                name = p.name.lower()
                if any(ext in name for ext in [".json", ".yaml", ".csv", ".txt", ".md"]) and not name.endswith(".jsonl"):
                    if any(k in name for k in ["eval", "bench", "result", "score", "araeval", "aramath", "arapro", "ien", "gsm8k", "math", "leaderboard"]):
                        if "checkpoint" not in str(p).lower() and "tokenizer" not in name and "adapter" not in name:
                            found_files.append((root, p))

print(f"Found {len(found_files)} potential benchmark/eval result files.\n")

for root, p in found_files:
    rel = p.relative_to(root).as_posix()
    print(f"📄 [{root.name} / {rel}]")
    try:
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            print(json.dumps(data, indent=2, ensure_ascii=False)[:1200])
        else:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            print(text[:1200])
    except Exception as e:
        print(f"   (Error reading file: {e})")
    print("-" * 70)
