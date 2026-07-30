import shutil
import json
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
outputs_dir = PACK_ROOT / "outputs"
backup_dir = outputs_dir / "saved_benchmark_samples"
backup_dir.mkdir(exist_ok=True)

files_to_save = [
    "model_a_deepseek_50.jsonl",
    "model_b_qwen37_50.jsonl",
    "model_b_deepseek_r1_50.jsonl",
    "benchmark_50_prompts.jsonl"
]

saved = []
for fname in files_to_save:
    src = outputs_dir / fname
    if src.exists():
        dst = backup_dir / fname
        shutil.copy(src, dst)
        lines = len(open(dst, encoding="utf-8").readlines())
        saved.append((fname, lines))
        print(f"[SAVED BACKUP] {fname} ({lines} records) -> {dst}")

# Also check for stage outputs under outputs/run/
run_dir = outputs_dir / "run"
if run_dir.exists():
    dst_run = backup_dir / "run_stages"
    if dst_run.exists():
        shutil.rmtree(dst_run)
    shutil.copytree(run_dir, dst_run)
    print(f"[SAVED BACKUP] Copying outputs/run stages -> {dst_run}")

print("\n" + "="*70)
print(" 📦 ALL GENERATED SAMPLES & STAGES PRESERVED IN SAFE BACKUP!")
print("="*70)
