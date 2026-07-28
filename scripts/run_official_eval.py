"""
Official Saudi-LLM Eval Team Runner Wrapper Script.
Executes C:\\Users\\Azooo\\Saudi-LLM\\Eval\\scripts\\run_araeval.py directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OFFICIAL_EVAL_SCRIPT = Path(r"C:\Users\Azooo\Saudi-LLM\Eval\scripts\run_araeval.py")


def main():
    parser = argparse.ArgumentParser(description="Run Official Saudi-LLM Eval Team Benchmark Engine")
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B", help="Base model name or path (default: unsloth/Qwen3.5-4B)")
    parser.add_argument("--adapter-path", help="Path to LoRA adapter directory (e.g., ./runs/sft_v1 or ./runs/grpo_master_5090)")
    parser.add_argument("--limit", type=int, help="Optional diagnostic per-task sample limit (e.g. 50)")
    parser.add_argument("--sample-size", type=int, help="Optional proportional sample size across all 7 tasks")
    parser.add_argument("--output-dir", default="./outputs/eval_team_results", help="Output directory for checkpoint.json and summary.json")
    args = parser.parse_args()

    if not OFFICIAL_EVAL_SCRIPT.exists():
        # Try relative path if running on Linux/WSL
        alt_script = Path("/workspace/Saudi-LLM/Eval/scripts/run_araeval.py")
        if alt_script.exists():
            eval_script_path = alt_script
        else:
            raise FileNotFoundError(f"Official evaluation script not found at {OFFICIAL_EVAL_SCRIPT} or {alt_script}")
    else:
        eval_script_path = OFFICIAL_EVAL_SCRIPT

    cmd = [
        sys.executable,
        str(eval_script_path),
        "--model", args.model,
        "--output-dir", args.output_dir,
    ]

    if args.adapter_path:
        cmd.extend(["--adapter-path", args.adapter_path])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    elif args.sample_size:
        cmd.extend(["--sample-size", str(args.sample_size)])

    print("=================================================================")
    print("LAUNCHING OFFICIAL SAUDI-LLM EVAL TEAM BENCHMARK ENGINE")
    print(f"Script: {eval_script_path}")
    print(f"Model : {args.model}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Output : {args.output_dir}")
    print("=================================================================\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
