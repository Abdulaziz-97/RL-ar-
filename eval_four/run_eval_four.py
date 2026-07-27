"""
Dedicated Evaluation Runner for Core 4 Benchmarks:
1. AraIFEval (Constraint Satisfaction)
2. AraPro (Accuracy)
3. AraTrust (Accuracy)
4. AraMath (Accuracy)

Usage:
    python eval_four/run_eval_four.py --model Qwen/Qwen3.5-4B --adapter ./runs/grpo_v1 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from araeval.config import AraEvalConfig, CORE_FOUR_DATASETS
from araeval.runner import AraEvalRunner


def main():
    parser = argparse.ArgumentParser(
        prog="eval_four",
        description="Run evaluation on the core 4 Arabic benchmarks: AraIFEval, AraPro, AraTrust, AraMath",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model name or path")
    parser.add_argument("--adapter", help="Path to SFT/GRPO LoRA adapter directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (0.6 standard for DeepSeekMath/Qwen2.5-Math evaluation)")
    parser.add_argument("--limit", type=int, help="Limit number of samples per task (optional)")
    parser.add_argument("--output", default="./araeval_results_four", help="Output directory")
    parser.add_argument("--tasks", nargs="+", default=CORE_FOUR_DATASETS, help="Tasks to evaluate (e.g. ara_math, ara_ifeval, ara_pro, ara_trust)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    args = parser.parse_args()

    selected_tasks = args.tasks

    print("=================================================================")
    print(f"RUNNING EVALUATION FOR TASKS: {', '.join(selected_tasks)}")
    print("=================================================================\n")

    cfg = AraEvalConfig(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        tasks=selected_tasks,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output,
        load_in_4bit=args.load_in_4bit,
    )

    runner = AraEvalRunner(cfg)
    summary = runner.run()

    print("\n=================================================================")
    print("CORE 4 BENCHMARK RESULTS SUMMARY:")
    print("=================================================================")
    for t_name in selected_tasks:
        t_data = summary["tasks"].get(t_name, {})
        acc = t_data.get("accuracy", 0.0) * 100
        n_samples = t_data.get("total_samples", 0)
        print(f"  - {t_name:<15}: {acc:6.2f}% ({t_data.get('correct_samples', 0)}/{n_samples} correct)")
    print(f"\n  Macro Average   : {summary.get('macro_accuracy', 0.0) * 100:6.2f}%")
    print("=================================================================")


if __name__ == "__main__":
    main()
