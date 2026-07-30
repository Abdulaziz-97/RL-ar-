import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

print("="*80)
print(" SEARCHING FOR ALL SCORE & EVALUATION METRIC FILES")
print("="*80)

for p in root.rglob("*.json"):
    if "checkpoint" not in str(p).lower() and "adapter_config" not in p.name:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rel = p.relative_to(root).as_posix()
            if isinstance(data, dict) and any(k in str(data).lower() for k in ["score", "acc", "accuracy", "loss", "bleu", "rouge", "eval"]):
                print(f"📄 [{rel}]")
                print(json.dumps(data, indent=2, ensure_ascii=False)[:600])
                print("-" * 60)
        except Exception:
            pass
