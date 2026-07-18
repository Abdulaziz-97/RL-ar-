"""Run SFT then GRPO from this pack (portable — no hardcoded machine paths)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n{'=' * 60}\nRUNNING: {' '.join(cmd)}\n{'=' * 60}", flush=True)
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
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        print(f"FAILED with code {p.returncode}", flush=True)
        sys.exit(p.returncode)
    print("\nDONE\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT → GRPO full pipeline")
    parser.add_argument("--sft-config", default="configs/qwen_4b_qlora.yaml")
    parser.add_argument("--grpo-config", default="configs/qwen_4b_smoke_v11.yaml")
    parser.add_argument("--sft-output", default="./runs/sft_v1")
    parser.add_argument("--grpo-output", default="./runs/grpo_v1")
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--grpo-max-steps", type=int, default=30)
    args = parser.parse_args()

    py = sys.executable

    run([
        py, "-u", "-m", "rlvr_pipeline", "sft",
        "--config", args.sft_config,
        "--output", args.sft_output,
        "--num-train-epochs", str(args.sft_epochs),
    ])

    run([
        py, "-u", "scripts/format_probe_sft.py",
        args.sft_output, "512",
    ])

    run([
        py, "-u", "-m", "rlvr_pipeline", "train",
        "--config", args.grpo_config,
        "--sft-checkpoint", args.sft_output,
        "--output", args.grpo_output,
        "--max-steps", str(args.grpo_max_steps),
    ])


if __name__ == "__main__":
    main()
