import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

print(f"Scanning root directory: {root}\n")

# List all top-level folders and files
top_items = sorted(list(root.glob("*")))
for item in top_items:
    print(f"📁 {item.name}")

print("\n" + "="*80)
print(" SEARCHING FOR EVALUATION REPORTS AND SCORES ACROSS ALL MODELS")
print("="*80)

eval_files = []
for p in root.rglob("*"):
    if p.is_file():
        name = p.name.lower()
        if any(k in name for k in ["eval", "report", "summary", "result", "metric", "score", "leaderboard", "bench"]):
            if not name.endswith(".bin") and not name.endswith(".safetensors") and "checkpoint" not in str(p).lower():
                eval_files.append(p)

print(f"Found {len(eval_files)} evaluation-related files:\n")
for f in eval_files[:35]:
    rel = f.relative_to(root)
    print(f"  📄 {rel}")
