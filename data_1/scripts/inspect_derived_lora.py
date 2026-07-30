import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B\derived-lora")

print("="*80)
print(" INSPECTING DERIVED-LORA ADAPTERS")
print("="*80)

for item in sorted(list(root.rglob("*"))):
    if item.is_dir() and (item / "adapter_config.json").exists():
        rel = item.relative_to(root).as_posix()
        cfg = json.loads((item / "adapter_config.json").read_text(encoding="utf-8"))
        print(f"🔹 [{rel}]")
        print(f"   r: {cfg.get('r')}, lora_alpha: {cfg.get('lora_alpha')}, target_modules: {cfg.get('target_modules')}")
        rep = item / "interpolation_report.json"
        if rep.exists():
            rep_data = json.loads(rep.read_text(encoding="utf-8"))
            print(f"   Interpolation: lambda_right={rep_data.get('weight_right')}, output_rank={rep_data.get('output_rank')}")
