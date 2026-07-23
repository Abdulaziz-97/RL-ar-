"""
CLI entry point for AraEval benchmark suite.

Usage:
    python -m araeval --model Qwen/Qwen3.5-4B --adapter ./runs/grpo_v1 --mode generation --tasks all
"""

from __future__ import annotations

import argparse
import sys

from araeval.config import AraEvalConfig
from araeval.runner import evaluate_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="araeval",
        description="AraEval: Arabic Multi-Task Evaluation Suite for LLMs",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Base model name or path")
    parser.add_argument("--adapter", help="Path to SFT/GRPO LoRA adapter directory")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks to evaluate (ara_ifeval, ara_truthfulqa, ara_math, ien_mcq, ien_tf, etec, lc_eval, or all)",
    )
    parser.add_argument(
        "--mode",
        choices=["generation", "loglik"],
        default="generation",
        help="Evaluation mode (generation or loglik)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--limit", type=int, help="Limit number of samples per task")
    parser.add_argument("--output", default="./araeval_results", help="Output directory")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = AraEvalConfig(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        tasks=args.tasks,
        eval_mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output,
        load_in_4bit=args.load_in_4bit,
        fp16=args.fp16,
        bf16=args.bf16 if not args.fp16 else False,
    )

    evaluate_model(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
