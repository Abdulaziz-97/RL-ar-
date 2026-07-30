import os
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

root = Path(r"I:\Instruction_following_team_handoff_extracted\Qwen3.5-4B")

models_data = []

for folder in sorted(list(root.glob("T*"))):
    if folder.is_dir():
        name = folder.name
        checkpoints = list((folder / "checkpoints").glob("checkpoint-*")) if (folder / "checkpoints").exists() else []
        
        best_loss = 999.0
        best_step = 0
        best_epoch = 0.0
        max_steps = 0
        lr = "1e-5" if "lr1e5" in name else "default"
        
        for ckpt in checkpoints:
            ts_path = ckpt / "trainer_state.json"
            if ts_path.exists():
                try:
                    data = json.loads(ts_path.read_text(encoding="utf-8"))
                    max_steps = data.get("max_steps", max_steps)
                    log_hist = data.get("log_history", [])
                    for entry in log_hist:
                        if "eval_loss" in entry:
                            el = entry["eval_loss"]
                            step = entry["step"]
                            epoch = entry.get("epoch", 0.0)
                            if el < best_loss:
                                best_loss = el
                                best_step = step
                                best_epoch = epoch
                except Exception:
                    pass

        models_data.append({
            "name": name,
            "best_loss": best_loss if best_loss != 999.0 else None,
            "best_step": best_step,
            "best_epoch": best_epoch,
            "max_steps": max_steps,
            "lr": lr,
            "ckpts": len(checkpoints)
        })

models_data.sort(key=lambda x: (x["best_loss"] if x["best_loss"] else 999.0))

print("="*95)
print(" 🏆 INSTRUCTION FOLLOWING TEAM OFFICIAL BENCHMARK LEADERBOARD (Qwen3.5-4B)")
print("="*95)
print(f"{'Rank':<5} | {'Model Variation':<42} | {'Eval Loss':<10} | {'Best Step':<10} | {'Epoch':<7} | {'Total Steps':<11} | {'Status'}")
print("-" * 95)

for rank, m in enumerate(models_data, 1):
    loss_str = f"{m['best_loss']:.4f}" if m['best_loss'] else "N/A"
    epoch_str = f"{m['best_epoch']:.2f}" if m['best_epoch'] else "N/A"
    status = "🥇 Best SOTA" if rank == 1 else ("🥈 Runner-up" if rank == 2 else ("🥉 Top 3" if rank == 3 else "Passed"))
    if "interrupted" in m['name']:
        status = "⚠️ Interrupted"
    print(f"{rank:<5} | {m['name']:<42} | {loss_str:<10} | {m['best_step']:<10} | {epoch_str:<7} | {m['max_steps']:<11} | {status}")

print("="*95)
