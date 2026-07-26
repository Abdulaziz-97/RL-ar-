"""
Dedicated Evaluation Runner for English Baseline Benchmarks:
1. IFEval (English Formatting & Structural Constraint Following)
2. MMLU (English Multitask Language Understanding)
3. HellaSwag (English Commonsense NLI & Completion)
4. GSM8K (English Mathematical CoT Reasoning)

Usage:
    python eval_english/run_eval_english.py --model Qwen/Qwen3.5-4B --adapter ./outputs/qwen_4b_rtx6000ada_run --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from english_eval.config import EnglishEvalConfig, DEFAULT_ENGLISH_TASKS
from english_eval.runner import EnglishEvalRunner


def main():
    parser = argparse.ArgumentParser(
        prog="eval_english",
        description="Run evaluation on standard English linguistic and reasoning benchmarks",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model name or path")
    parser.add_argument("--adapter", help="Path to SFT/GRPO LoRA adapter directory")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Tasks to evaluate (ifeval, mmlu, hellaswag, gsm8k, all)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--limit", type=int, help="Limit number of samples per task (optional)")
    parser.add_argument("--output", default="./english_eval_results", help="Output directory")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    args = parser.parse_args()

    print("=================================================================")
    print("RUNNING EVALUATION FOR ENGLISH BASELINE BENCHMARKS:")
    print("   1. IFEval      (google/ifeval - Constraint Adherence)")
    print("   2. MMLU        (cais/mmlu - Language Understanding)")
    print("   3. HellaSwag   (Rowan/hellaswag - Commonsense NLI)")
    print("   4. GSM8K       (gsm8k - English CoT Math Reasoning)")
    print("=================================================================\n")

    cfg = EnglishEvalConfig(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        tasks=args.tasks,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output,
        load_in_4bit=args.load_in_4bit,
    )

    runner = EnglishEvalRunner(cfg)
    summary = runner.run()

    print("\n=================================================================")
    print("ENGLISH BENCHMARK RESULTS SUMMARY:")
    print("=================================================================")
    task_list = cfg.get_task_list()
    for t_name in task_list:
        t_data = summary["tasks"].get(t_name, {})
        acc = t_data.get("accuracy", 0.0) * 100
        n_samples = t_data.get("total_samples", 0)
        print(f"  - {t_name:<15}: {acc:6.2f}% ({t_data.get('correct_samples', 0)}/{n_samples} correct)")
    print(f"\n  Macro Average   : {summary.get('macro_accuracy', 0.0) * 100:6.2f}%")
    print("=================================================================")


if __name__ == "__main__":
    main()
