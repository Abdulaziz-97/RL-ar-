"""
Saudi-LLM Master V4 Orchestrator — verifier-first datagen + training.

Order when REGENERATE_DATA=1:
  SFT generate → select 4k → SFT train → RLVR generate → pass@8 → curate 4k →
  dual ship gate / promote → GRPO → AraEval

Normal training consumes the immutable promoted release only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PACK_ROOT.parent

CONFIG_FILE = ROOT_DIR / "configs" / "qwen_4b_2x5090_v4_sota.yaml"
SFT_OUT = ROOT_DIR / "outputs" / "sft_coldstart_v4"
GRPO_OUT = ROOT_DIR / "outputs" / "qwen_4b_2x5090_v4_run"
EVAL_OUT = ROOT_DIR / "outputs" / "generative_eval_grpo_v4"


def run_cmd(cmd: str, check: bool = True) -> None:
    print(f"\n[Master Command] Executing: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        print(f"Command failed with return code {res.returncode}: {cmd}", flush=True)
        sys.exit(res.returncode)


def main() -> None:
    print("=" * 85, flush=True)
    print(" SAUDI-LLM V4 MASTER ORCHESTRATOR", flush=True)
    print("=" * 85, flush=True)

    try:
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
        num_gpus = len([l for l in gpu_info.strip().split("\n") if l.strip()])
    except Exception:
        num_gpus = 1
    print(f"[System Diagnostic] Detected {num_gpus} active NVIDIA GPU(s).", flush=True)

    regenerate = os.environ.get("REGENERATE_DATA", "0") == "1"
    if regenerate:
        print("REGENERATE_DATA=1 — delegating to setup_and_run_vastai.sh --fg", flush=True)
        env = os.environ.copy()
        env["REGENERATE_DATA"] = "1"
        cmd = f"bash {ROOT_DIR / 'scripts' / 'setup_and_run_vastai.sh'} --fg"
        res = subprocess.run(cmd, shell=True, env=env)
        sys.exit(res.returncode)

    print("\n[1] Dual-corpus ship gate (immutable release)", flush=True)
    run_cmd(f"python {SCRIPT_DIR}/verify_master_v4_datasets_complete.py")

    print("\n[2] Cold-start SFT", flush=True)
    if num_gpus > 1:
        sft_cmd = (
            f"torchrun --nproc_per_node={num_gpus} -m rlvr_pipeline.cli sft "
            f"--config {CONFIG_FILE} --output {SFT_OUT}"
        )
    else:
        sft_cmd = f"python -m rlvr_pipeline.cli sft --config {CONFIG_FILE} --output {SFT_OUT}"
    run_cmd(sft_cmd)
    run_cmd(f"python -m rlvr_pipeline.checkpoint_integrity --checkpoint {SFT_OUT} --require-adapter-only")

    print("\n[3] GRPO", flush=True)
    sft_arg = f"--sft-checkpoint {SFT_OUT}"
    if num_gpus > 1:
        grpo_cmd = (
            f"torchrun --nproc_per_node={num_gpus} -m rlvr_pipeline.cli train "
            f"--config {CONFIG_FILE} --output {GRPO_OUT} {sft_arg}"
        )
    else:
        grpo_cmd = (
            f"python -m rlvr_pipeline.cli train --config {CONFIG_FILE} "
            f"--output {GRPO_OUT} {sft_arg}"
        )
    run_cmd(grpo_cmd)

    ckpt_candidates = list(GRPO_OUT.glob("checkpoint-*"))
    latest_ckpt = max(ckpt_candidates, key=lambda p: p.stat().st_mtime) if ckpt_candidates else GRPO_OUT
    run_cmd(
        f"python -m rlvr_pipeline.checkpoint_integrity --checkpoint {latest_ckpt} --require-adapter-only"
    )

    print("\n[4] AraEval", flush=True)
    eval_cmd = (
        f"python {ROOT_DIR}/official_eval/run_araeval_generative.py "
        f"--model Qwen/Qwen3.5-4B "
        f"--instruction-adapter aziz9788/T06__qwen35-mixed-v6-lr1e5 "
        f"--adapter-path {latest_ckpt} --enable-thinking --engine hf "
        f"--output-dir {EVAL_OUT}"
    )
    run_cmd(eval_cmd)
    print("V4 master orchestrator complete.", flush=True)


if __name__ == "__main__":
    main()
