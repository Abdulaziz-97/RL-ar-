"""
High-Speed Side-by-Side Evaluation Script:
Evaluates Base Model (Qwen/Qwen3.5-4B) vs GRPO_V2 (aziz9788/qwen3.5-4b-arabic-grpo-v2)
on Core 4 Benchmarks with high batch-size throughput (batch_size=64).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from araeval.config import AraEvalConfig, CORE_FOUR_DATASETS
from araeval.runner import AraEvalRunner


def eval_single_model(model_name: str, adapter_path: str | None, batch_size: int, output_dir: str, limit: int | None = None) -> dict:
    print(f"\n=================================================================")
    print(f"EVALUATING MODEL: {adapter_path or model_name}")
    print(f"Batch Size: {batch_size} | High-Speed Vectorized Generation")
    print("=================================================================\n")

    cfg = AraEvalConfig(
        model_name_or_path=model_name,
        adapter_path=adapter_path,
        tasks=CORE_FOUR_DATASETS,
        batch_size=batch_size,
        max_new_tokens=512,
        temperature=0.0,  # Greedy deterministic evaluation for maximum accuracy & speed
        limit=limit,
        output_dir=output_dir,
    )
    runner = AraEvalRunner(cfg)
    return runner.run()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base Model vs GRPO_V2 at Maximum Speed")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size (default: 64 for maximum speed)")
    parser.add_argument("--grpo-adapter", default="aziz9788/qwen3.5-4b-arabic-grpo-v2", help="Hugging Face HF repo or local adapter path for GRPO_V2")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples per task (for testing)")
    args = parser.parse_args()

    batch_size = args.batch_size
    grpo_adapter = args.grpo_adapter
    limit = args.limit

    # 1. Evaluate Base Model (Qwen/Qwen3.5-4B)
    base_summary = eval_single_model(
        model_name="Qwen/Qwen3.5-4B",
        adapter_path=None,
        batch_size=batch_size,
        limit=limit,
        output_dir="./outputs/eval_base_model",
    )

    # 2. Evaluate GRPO_V2 Model (aziz9788/qwen3.5-4b-arabic-grpo-v2)
    grpo_summary = eval_single_model(
        model_name="Qwen/Qwen3.5-4B",
        adapter_path=grpo_adapter,
        batch_size=batch_size,
        limit=limit,
        output_dir="./outputs/eval_grpo_v2",
    )

    # 3. Print Side-by-Side Comparison Table
    print("\n\n" + "=" * 70)
    print("      CORE 4 BENCHMARK SIDE-BY-SIDE COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Task / Benchmark':<18} | {'Base (Qwen3.5-4B)':<18} | {'GRPO_V2 (Our Model)':<18} | {'Delta':<8}")
    print("-" * 70)

    for task in CORE_FOUR_DATASETS:
        base_acc = base_summary["tasks"].get(task, {}).get("accuracy", 0.0) * 100
        grpo_acc = grpo_summary["tasks"].get(task, {}).get("accuracy", 0.0) * 100
        delta = grpo_acc - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"{task:<18} | {base_acc:6.2f}%            | {grpo_acc:6.2f}%            | {sign}{delta:5.2f}%")

    base_macro = base_summary.get("macro_accuracy", 0.0) * 100
    grpo_macro = grpo_summary.get("macro_accuracy", 0.0) * 100
    macro_delta = grpo_macro - base_macro
    macro_sign = "+" if macro_delta >= 0 else ""

    print("-" * 70)
    print(f"{'MACRO AVERAGE':<18} | {base_macro:6.2f}%            | {grpo_macro:6.2f}%            | {macro_sign}{macro_delta:5.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
