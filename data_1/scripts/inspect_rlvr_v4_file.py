import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
root_dir = PACK_ROOT.parent

rlvr_path = root_dir / "data" / "arabic_reasoning_rlvr_v4.jsonl"

print("="*80)
print(f" INSPECTING RLVR V4 FILE: {rlvr_path}")
print("="*80)

if not rlvr_path.exists():
    print("❌ File does not exist.")
    sys.exit(0)

mtime = datetime.fromtimestamp(os.path.getmtime(rlvr_path)).strftime('%Y-%m-%d %H:%M:%S')
size_mb = os.path.getsize(rlvr_path) / (1024 * 1024)

with open(rlvr_path, encoding="utf-8") as f:
    items = [json.loads(l) for l in f if l.strip()]

print(f"File Modified Time: {mtime}")
print(f"File Size:         {size_mb:.2f} MB")
print(f"Total Prompt Rows: {len(items)}")

if items:
    print("\n--- SAMPLE 1 ---")
    print(json.dumps(items[0], indent=2, ensure_ascii=False))
    print("\n--- SAMPLE 4000 ---")
    print(json.dumps(items[-1], indent=2, ensure_ascii=False))
