import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

results = []

for folder in sorted(list(root.glob("*"))):
    if folder.is_dir() and folder.name.startswith("T"):
        name = folder.name
        checkpoints = list((folder / "checkpoints").glob("checkpoint-*")) if (folder / "checkpoints").exists() else []
        
        best_loss = 999.0
        best_step = None
        final_loss = None
        
        for ckpt in checkpoints:
            ts_path = ckpt / "trainer_state.json"
            if ts_path.exists():
                try:
                    data = json.loads(ts_path.read_text(encoding="utf-8"))
                    log_hist = data.get("log_history", [])
                    for entry in log_hist:
                        if "eval_loss" in entry:
                            el = entry["eval_loss"]
                            step = entry["step"]
                            if el < best_loss:
                                best_loss = el
                                best_step = step
                            final_loss = el
                except Exception as e:
                    pass
        
        results.append({
            "model": name,
            "best_eval_loss": best_loss if best_loss != 999.0 else None,
            "best_step": best_step,
            "final_eval_loss": final_loss,
            "checkpoints_count": len(checkpoints)
        })

print("="*85)
print(" INSTRUCTION FOLLOWING TEAM HANDOFF EVALUATION SUMMARY (Qwen3.5-4B)")
print("="*85)
print(f"{'Model / Experiment':<45} | {'Best Eval Loss':<15} | {'Best Step':<10} | {'Checkpoints'}")
print("-" * 85)

for r in sorted(results, key=lambda x: (x['best_eval_loss'] if x['best_eval_loss'] else 999)):
    bl = f"{r['best_eval_loss']:.4f}" if r['best_eval_loss'] else "N/A"
    bs = str(r['best_step']) if r['best_step'] else "N/A"
    print(f"{r['model']:<45} | {bl:<15} | {bs:<10} | {r['checkpoints_count']}")
