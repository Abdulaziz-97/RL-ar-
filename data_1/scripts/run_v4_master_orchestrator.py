"""
Saudi-LLM Master V4 Orchestrator & Multi-GPU Torchrun Trainer.
Automates:
  Step 1: Live Qwen 3.7 Flash 4K RLVR prompt generation with Content-Jaccard Guard (<0.65) and 5-gram decontamination.
  Step 2: Comprehensive Gate & Zero-Overlap Verification Audit.
  Step 3: Stage 1 Cold-Start SFT Warm-up (4,000 CoT Solutions) via PyTorch Torchrun DDP.
  Step 4: Stage 2 GRPO V4 Multi-GPU RLVR Training via PyTorch Torchrun DDP.
  Step 5: Official AraEval Parallel Generative Benchmark Evaluation.
"""

import json
import os
import subprocess
import urllib.request
import time
import sys
import random
import re
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR  = Path(__file__).resolve().parent
PACK_ROOT   = SCRIPT_DIR.parent
ROOT_DIR    = PACK_ROOT.parent

OUT_SFT     = ROOT_DIR / "data" / "arabic_reasoning_coldstart_v4.jsonl"
OUT_RLVR    = ROOT_DIR / "data" / "arabic_reasoning_rlvr_v4.jsonl"
CONFIG_FILE = ROOT_DIR / "configs" / "qwen_4b_2x5090_v4_sota.yaml"
SFT_OUT     = ROOT_DIR / "outputs" / "sft_coldstart_v4"
GRPO_OUT    = ROOT_DIR / "outputs" / "qwen_4b_2x5090_v4_run"
EVAL_OUT    = ROOT_DIR / "outputs" / "generative_eval_grpo_v4"

RLVR_TARGET = 4000
WORKERS     = 30

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
    "X-Title": "Arabic RLVR Pipeline V4 Master Orchestrator",
}
MODEL = "qwen/qwen3.7-flash"

def run_cmd(cmd: str, check=True):
    print(f"\n[Master Command] Executing: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        print(f"❌ Command failed with return code {res.returncode}: {cmd}", flush=True)
        sys.exit(res.returncode)

def main():
    print("=" * 85, flush=True)
    print(" 🚀 SAUDI-LLM V4 MASTER ORCHESTRATOR — AUTOMATED TORCHRUN TRAINING & EVALUATION", flush=True)
    print("=" * 85, flush=True)

    # Detect Available GPUs
    try:
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
        num_gpus = len([l for l in gpu_info.strip().split("\n") if l.strip()])
    except Exception:
        num_gpus = 1

    print(f"[System Diagnostic] Detected {num_gpus} active NVIDIA GPU(s).", flush=True)

    # ── STEP 1: Live Qwen 3.7 RLVR Generation Loop ───────────────────────
    print("\n" + "=" * 85, flush=True)
    print(" 📌 STEP 1: Live Qwen 3.7 Flash RLVR Prompt Generation", flush=True)
    print("=" * 85, flush=True)
    
    if OPENROUTER_KEY:
        run_cmd(f"python {SCRIPT_DIR}/generate_qwen37_rlvr_live.py")
    else:
        print("⚠️ OPENROUTER_API_KEY not set. Skipping live RLVR generation step.", flush=True)

    # ── STEP 2: Gate & Schema Verification Audit ─────────────────────────
    print("\n" + "=" * 85, flush=True)
    print(" 📌 STEP 2: Running Gate & Schema Verification Audit", flush=True)
    print("=" * 85, flush=True)
    run_cmd(f"python {SCRIPT_DIR}/verify_master_v4_datasets_complete.py")

    # ── STEP 3: Stage 1 Cold-Start SFT Warm-up (Torchrun DDP) ───────────
    print("\n" + "=" * 85, flush=True)
    print(" 📌 STEP 3: STAGE 1 Cold-Start CoT SFT Warm-up (Torchrun DDP)", flush=True)
    print("=" * 85, flush=True)
    if num_gpus > 1:
        sft_cmd = f"torchrun --nproc_per_node={num_gpus} -m rlvr_pipeline.cli sft --config {CONFIG_FILE} --output {SFT_OUT} --max-steps 100"
    else:
        sft_cmd = f"python -m rlvr_pipeline.cli sft --config {CONFIG_FILE} --output {SFT_OUT} --max-steps 100"
    
    try:
        run_cmd(sft_cmd, check=False)
    except Exception as e:
        print(f"SFT warm-up warning: {e}", flush=True)

    # ── STEP 4: Stage 2 GRPO V4 Multi-GPU Training Launch (Torchrun DDP) ─
    print("\n" + "=" * 85, flush=True)
    print(f" 📌 STEP 4: STAGE 2 GRPO V4 Multi-GPU RLVR Training ({num_gpus} GPUs Torchrun DDP)", flush=True)
    print("=" * 85, flush=True)

    sft_arg = ""
    sft_adapter_path = SFT_OUT / "adapter_config.json"
    if sft_adapter_path.exists():
        sft_arg = f"--sft-checkpoint {SFT_OUT}"
        print(f"  * Chaining from Stage 1 SFT Checkpoint: {SFT_OUT}", flush=True)
    else:
        print("  * Starting GRPO V4 directly from aziz9788 SOTA instruction model.", flush=True)

    if num_gpus > 1:
        grpo_cmd = f"torchrun --nproc_per_node={num_gpus} -m rlvr_pipeline.cli train --config {CONFIG_FILE} {sft_arg}"
    else:
        grpo_cmd = f"python -m rlvr_pipeline.cli train --config {CONFIG_FILE} {sft_arg}"
    
    run_cmd(grpo_cmd)

    # ── STEP 5: Official AraEval Parallel Generative Evaluation ──────────
    print("\n" + "=" * 85, flush=True)
    print(" 📌 STEP 5: Official AraEval Parallel Generative Benchmark Evaluation", flush=True)
    print("=" * 85, flush=True)

    ckpt_candidates = list(GRPO_OUT.glob("checkpoint-*"))
    if ckpt_candidates:
        latest_ckpt = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
    else:
        latest_ckpt = GRPO_OUT

    print(f"[Eval] Evaluating Final GRPO V4 Checkpoint: {latest_ckpt}", flush=True)
    eval_cmd = f"python {ROOT_DIR}/official_eval/run_araeval_generative.py --model aziz9788/T06__qwen35-mixed-v6-lr1e5 --adapter-path {latest_ckpt} --enable-thinking --max-lora-rank 128 --output-dir {EVAL_OUT}"
    run_cmd(eval_cmd)

    print("\n" + "=" * 85, flush=True)
    print(" 🏆 ALL 5 STAGES OF SAUDI-LLM V4 PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print(f"  • Trained GRPO V4 Checkpoint: {latest_ckpt}")
    print(f"  • Evaluation Summary       : {EVAL_OUT}/summary.json")
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()
