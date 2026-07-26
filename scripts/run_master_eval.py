"""
Master Evaluation Suite: Evaluates Base Qwen3.5-4B vs. Trained GRPO Model
across all 4 Arabic Core Benchmarks + 4 English Baseline Benchmarks.

Usage:
    python scripts/run_master_eval.py --adapter /workspace/RL-ar-/runs/grpo_v1 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is in python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from araeval.config import AraEvalConfig, CORE_FOUR_DATASETS
from araeval.runner import AraEvalRunner
from english_eval.config import EnglishEvalConfig, DEFAULT_ENGLISH_TASKS
from english_eval.runner import EnglishEvalRunner


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(base_ara: dict, trained_ara: dict, base_eng: dict, trained_eng: dict):
    print("\n" + "=" * 90)
    print("      MASTER EVALUATION MATRIX: BASE MODEL vs. TRAINED RLVR MODEL")
    print("=" * 90)
    print(f"{'Benchmark / Category':<35} | {'Base Model':<14} | {'Trained Model':<14} | {'Delta':<10}")
    print("-" * 90)

    print(" [ARABIC CORE 4 BENCHMARKS]")
    for t_name in CORE_FOUR_DATASETS:
        b_acc = base_ara.get("tasks", {}).get(t_name, {}).get("accuracy", 0.0) * 100
        t_acc = trained_ara.get("tasks", {}).get(t_name, {}).get("accuracy", 0.0) * 100
        delta = t_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print(f"   - {t_name:<30} | {b_acc:6.2f}%       | {t_acc:6.2f}%       | {sign}{delta:6.2f}%")

    b_ara_macro = base_ara.get("macro_accuracy", 0.0) * 100
    t_ara_macro = trained_ara.get("macro_accuracy", 0.0) * 100
    ara_delta = t_ara_macro - b_ara_macro
    sign_ara = "+" if ara_delta >= 0 else ""
    print("-" * 90)
    print(f"   * ARABIC MACRO AVERAGE            | {b_ara_macro:6.2f}%       | {t_ara_macro:6.2f}%       | {sign_ara}{ara_delta:6.2f}%")
    print("=" * 90)

    print(" [ENGLISH BASELINE BENCHMARKS]")
    for t_name in DEFAULT_ENGLISH_TASKS:
        b_acc = base_eng.get("tasks", {}).get(t_name, {}).get("accuracy", 0.0) * 100
        t_acc = trained_eng.get("tasks", {}).get(t_name, {}).get("accuracy", 0.0) * 100
        delta = t_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print(f"   - {t_name:<30} | {b_acc:6.2f}%       | {t_acc:6.2f}%       | {sign}{delta:6.2f}%")

    b_eng_macro = base_eng.get("macro_accuracy", 0.0) * 100
    t_eng_macro = trained_eng.get("macro_accuracy", 0.0) * 100
    eng_delta = t_eng_macro - b_eng_macro
    sign_eng = "+" if eng_delta >= 0 else ""
    print("-" * 90)
    print(f"   * ENGLISH MACRO AVERAGE           | {b_eng_macro:6.2f}%       | {t_eng_macro:6.2f}%       | {sign_eng}{eng_delta:6.2f}%")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Full Arabic + English Comparison Benchmark Suite")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model name or path")
    parser.add_argument("--adapter", default="/workspace/RL-ar-/runs/grpo_v1", help="Path to trained LoRA adapter")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--limit", type=int, help="Optional sample limit for fast testing")
    parser.add_argument("--output-dir", default="./master_eval_results", help="Output directory")
    args = parser.parse_args()

    # Prevent HuggingFace cache locking / stale file handle issues
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    output_base = Path(args.output_dir) / "base"
    output_trained = Path(args.output_dir) / "trained"

    print("\n=================================================================")
    print("STEP 1/4: EVALUATING BASE MODEL ON ARABIC CORE 4 BENCHMARKS")
    print("=================================================================")
    ara_cfg_base = AraEvalConfig(
        model_name_or_path=args.model,
        adapter_path=None,
        tasks=CORE_FOUR_DATASETS,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=str(output_base / "arabic"),
    )
    runner_ara_base = AraEvalRunner(ara_cfg_base)
    base_ara_results = runner_ara_base.run()

    print("\n=================================================================")
    print("STEP 2/4: EVALUATING BASE MODEL ON ENGLISH BASELINE BENCHMARKS")
    print("=================================================================")
    eng_cfg_base = EnglishEvalConfig(
        model_name_or_path=args.model,
        adapter_path=None,
        tasks=DEFAULT_ENGLISH_TASKS,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=str(output_base / "english"),
    )
    runner_eng_base = EnglishEvalRunner(eng_cfg_base)
    base_eng_results = runner_eng_base.run()

    print("\n=================================================================")
    print("STEP 3/4: EVALUATING TRAINED MODEL ON ARABIC CORE 4 BENCHMARKS")
    print("=================================================================")
    ara_cfg_trained = AraEvalConfig(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        tasks=CORE_FOUR_DATASETS,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=str(output_trained / "arabic"),
    )
    runner_ara_trained = AraEvalRunner(ara_cfg_trained)
    trained_ara_results = runner_ara_trained.run()

    print("\n=================================================================")
    print("STEP 4/4: EVALUATING TRAINED MODEL ON ENGLISH BASELINE BENCHMARKS")
    print("=================================================================")
    eng_cfg_trained = EnglishEvalConfig(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        tasks=DEFAULT_ENGLISH_TASKS,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=str(output_trained / "english"),
    )
    runner_eng_trained = EnglishEvalRunner(eng_cfg_trained)
    trained_eng_results = runner_eng_trained.run()

    print_comparison_table(
        base_ara_results, trained_ara_results, base_eng_results, trained_eng_results
    )


if __name__ == "__main__":
    main()
