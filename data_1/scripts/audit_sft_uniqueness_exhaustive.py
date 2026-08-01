import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
root_dir = PACK_ROOT.parent

sft_path = root_dir / "data" / "arabic_reasoning_coldstart_v5.jsonl"

print("="*80)
print(" 🔬 EXHAUSTIVE SFT DATASET UNIQUENESS AUDIT")
print(f" Target File: {sft_path}")
print("="*80)

if not sft_path.exists():
    print("❌ File does not exist yet.")
    sys.exit(1)

with open(sft_path, encoding="utf-8") as f:
    items = [json.loads(l) for l in f if l.strip()]

total_samples = len(items)
print(f"Total SFT Samples in File: {total_samples}")

# 1. Exact string deduplication
exact_prompts = [item.get("prompt", "") for item in items]
unique_exact = set(exact_prompts)
exact_dupes = total_samples - len(unique_exact)

# 2. Whitespace-normalized deduplication
norm_prompts = [" ".join(p.strip().split()) for p in exact_prompts]
unique_norm = set(norm_prompts)
norm_dupes = total_samples - len(unique_norm)

print(f"\n📊 AUDIT RESULTS:")
print(f"  • Total Rows: {total_samples}")
print(f"  • Unique Exact Prompts: {len(unique_exact)} (Duplicates: {exact_dupes})")
print(f"  • Unique Normalized Prompts: {len(unique_norm)} (Duplicates: {norm_dupes})")

if norm_dupes > 0:
    print(f"\n⚠️ FOUND {norm_dupes} DUPLICATES IN SFT DATASET!")
    # Find duplicate prompts
    seen = set()
    dupe_prompts = set()
    for p in norm_prompts:
        if p in seen:
            dupe_prompts.add(p)
        seen.add(p)
    
    print("\nSample Duplicates:")
    for dp in list(dupe_prompts)[:3]:
        print(f"  - '{dp[:100]}...'")
else:
    print("\n✅ 100% PERFECT UNIQUENESS VERIFIED — 0 DUPLICATE PROMPTS DETECTED!")
