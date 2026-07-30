import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

print("="*80)
print(" RECURSIVE FILE LISTING FOR ALL MODEL DIRECTORIES")
print("="*80)

for folder in sorted(list(root.glob("*"))):
    if folder.is_dir():
        print(f"\n📂 [{folder.name}]")
        files = list(folder.rglob("*"))
        subdirs = [f for f in files if f.is_dir()]
        regular_files = [f for f in files if f.is_file()]
        print(f"   Subdirectories ({len(subdirs)}): {[s.relative_to(folder).as_posix() for s in subdirs[:10]]}")
        print(f"   Files ({len(regular_files)}): {[f.relative_to(folder).as_posix() for f in regular_files[:15]]}")
