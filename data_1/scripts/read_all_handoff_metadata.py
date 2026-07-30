import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

print("="*80)
print(" READING ALL README.MD AND TRAINER_STATE.JSON FILES")
print("="*80)

# Check all README.md files
readmes = list(root.rglob("README.md"))
print(f"Found {len(readmes)} README.md files.\n")

for r in readmes:
    rel = r.relative_to(root).as_posix()
    content = r.read_text(encoding="utf-8", errors="ignore").strip()
    if content and len(content) > 20 and "model-index" not in content and "datasets:" not in content:
        print(f"--- 📄 {rel} ---")
        print(content[:1500])
        print("\n")

# Check all trainer_state.json files
trainer_states = list(root.rglob("trainer_state.json"))
print(f"\nFound {len(trainer_states)} trainer_state.json files.\n")
for ts in trainer_states:
    rel = ts.relative_to(root).as_posix()
    data = json.loads(ts.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])
    eval_logs = [l for l in log_history if "eval_loss" in l or "eval_accuracy" in l or "eval_runtime" in l]
    print(f"--- 📊 {rel} (Total steps: {data.get('max_steps')}, best_model: {data.get('best_model_checkpoint')}) ---")
    if eval_logs:
        for el in eval_logs[-3:]:
            print("  ", el)
    elif log_history:
        print("   Last log:", log_history[-1])
    print()

# Check all interpolation_report.json files
reports = list(root.rglob("interpolation_report.json"))
print(f"\nFound {len(reports)} interpolation_report.json files.\n")
for rep in reports:
    rel = rep.relative_to(root).as_posix()
    data = json.loads(rep.read_text(encoding="utf-8"))
    print(f"--- 📈 {rel} ---")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
    print()
