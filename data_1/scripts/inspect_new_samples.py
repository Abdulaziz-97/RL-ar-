import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
root_dir = PACK_ROOT.parent

sft_path = root_dir / "data" / "arabic_reasoning_coldstart_v4.jsonl"

with open(sft_path, encoding="utf-8") as f:
    items = [json.loads(l) for l in f if l.strip()]

new_items = items[1610:]
print(f"=== TOTAL NEW QWEN 3.7 SAMPLES SAVED: {len(new_items)} ===\n")

for i, item in enumerate(new_items[:5]):
    print(f"========================================================================")
    print(f"📌 [SAMPLE {i+1}] ID: {item.get('problem_id')} | Domain: {item.get('domain')}")
    print(f"========================================================================")
    print(f"📝 PROMPT:\n{item.get('prompt')}\n")
    print(f"🧠 RESPONSE (CoT):\n{item.get('response')}\n")
    print(f"🎯 GOLD: {item.get('metadata', {}).get('ground_truth_answer')}")
    print(f"⚙️ ANSWER SPEC: {json.dumps(item.get('answer_spec'), ensure_ascii=False)}")
    print("\n")
