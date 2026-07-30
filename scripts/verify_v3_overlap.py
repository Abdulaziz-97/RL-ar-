import json
from pathlib import Path

root = Path(__file__).resolve().parents[0]

sft_path = root / "data" / "arabic_reasoning_coldstart_train.jsonl"
rlvr_path = root / "data" / "arabic_reasoning_rlvr_train.jsonl"

sft_items = [json.loads(l) for l in open(sft_path, encoding="utf-8") if l.strip()]
rlvr_items = [json.loads(l) for l in open(rlvr_path, encoding="utf-8") if l.strip()]

sft_prompts = [item.get("prompt", "").strip() for item in sft_items]
rlvr_prompts = [item.get("prompt", "").strip() for item in rlvr_items]

set_sft = set(sft_prompts)
set_rlvr = set(rlvr_prompts)

overlap = set_sft & set_rlvr

print("="*80)
print(" 🔍 EMPIRICAL AUDIT: V3 SFT vs V3 RLVR PROMPT OVERLAP CHECK")
print("="*80)
print(f" Total SFT Samples in '{sft_path.name}':     {len(sft_prompts)}")
print(f" Total RLVR Prompts in '{rlvr_path.name}':     {len(rlvr_prompts)}")
print(f" 🚨 EXACT PROMPT MATCH OVERLAP:                    {len(overlap)} / {len(sft_prompts)} ({len(overlap)/len(sft_prompts)*100:.1f}%)")
print("="*80)

print("\n📌 FIRST 3 SFT PROMPTS:")
for i in range(3):
    print(f"  [{i+1}] {sft_prompts[i][:80]}...")

print("\n📌 FIRST 3 RLVR PROMPTS:")
for i in range(3):
    print(f"  [{i+1}] {rlvr_prompts[i][:80]}...")
print("="*80)
