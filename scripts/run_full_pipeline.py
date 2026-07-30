"""Run SFT -> GRPO -> Core 4 Eval in one end-to-end automated pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_run.log"
    
    header = f"\n{'=' * 60}\nRUNNING STAGE: {' '.join(cmd)}\n{'=' * 60}\n"
    print(header, flush=True)
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write(header)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    with open(log_file, "a", encoding="utf-8") as f_log:
        for line in p.stdout:
            print(line, end="", flush=True)
            f_log.write(line)
            f_log.flush()
    p.wait()
    if p.returncode != 0:
        msg = f"\nSTAGE FAILED with code {p.returncode}\n"
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(msg)
        sys.exit(p.returncode)
    msg = "\nSTAGE COMPLETED SUCCESSFULLY\n"
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Arabic Reasoning Pipeline: SFT -> GRPO -> Core4 Eval")
    parser.add_argument("--config", default="configs/qwen_4b_2xA40_production.yaml", help="Master YAML configuration file")
    parser.add_argument("--sft-output", default="./runs/sft_v1", help="Output directory for SFT checkpoint")
    parser.add_argument("--grpo-output", default="./runs/grpo_v1", help="Output directory for GRPO checkpoint")
    parser.add_argument("--eval-output", default="./outputs/eval_four_results", help="Output directory for Core 4 evaluation")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT phase if checkpoint already exists")
    parser.add_argument("--skip-eval", action="store_true", help="Skip Core 4 evaluation stage")
    args = parser.parse_args()

    py = sys.executable

    # 1. Phase 1: Cold-Start SFT
    if not args.skip_sft:
        run([
            py, "-u", "-m", "rlvr_pipeline", "sft",
            "--config", args.config,
            "--output", args.sft_output,
        ])

    # 2. Phase 1.5: SFT Format Probe Verification
    if (ROOT / "scripts" / "format_probe_sft.py").exists() and (Path(args.sft_output)).exists():
        try:
            run([
                py, "-u", "scripts/format_probe_sft.py",
                args.sft_output, "512",
            ])
        except SystemExit:
            print("WARNING: Format probe failed or warned, continuing to GRPO...", flush=True)

    # 3. Phase 2: GRPO RLVR Reinforcement Learning
    run([
        py, "-u", "-m", "rlvr_pipeline", "train",
        "--config", args.config,
        "--sft-checkpoint", args.sft_output,
        "--output", args.grpo_output,
    ])

    # 4. Phase 3: Core 4 Benchmark Evaluation (AraIFEval, AraPro, AraTrust, AraMath)
    if not args.skip_eval:
        run([
            py, "-u", "eval_four/run_eval_four.py",
            "--model", "Qwen/Qwen3.5-4B",
            "--adapter", args.grpo_output,
            "--output", args.eval_output,
        ])


if __name__ == "__main__":
    main()
