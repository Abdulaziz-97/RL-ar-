#!/bin/bash
# High-Speed Official Saudi-LLM Eval Team Benchmark Script for Vast.ai GPU Instances
set -e

# Ensure HF_TOKEN is set in environment (e.g. export HF_TOKEN="hf_...")
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export VLLM_USE_FLASHINFER_SAMPLER="0"

EVAL_SCRIPT_PATH="/workspace/Saudi-LLM/Eval/scripts/run_araeval.py"
if [ ! -f "$EVAL_SCRIPT_PATH" ]; then
    EVAL_SCRIPT_PATH="./Saudi-LLM/Eval/scripts/run_araeval.py"
fi

echo "================================================================="
echo "LAUNCHING OFFICIAL BENCHMARKS ON VAST.AI GPU INSTANCE"
echo "Eval Script: $EVAL_SCRIPT_PATH"
echo "================================================================="

# Step 1: Run Official Benchmark on Base Model (unsloth/Qwen3.5-4B)
echo ""
echo "[1/2] Evaluating Base Model (unsloth/Qwen3.5-4B)..."
python "$EVAL_SCRIPT_PATH" \
  --model unsloth/Qwen3.5-4B \
  --output-dir ./outputs/official_eval_base_model

# Step 2: Run Official Benchmark on GRPO_V2 (aziz9788/qwen3.5-4b-arabic-grpo-v2)
echo ""
echo "[2/2] Evaluating GRPO_V2 Model (aziz9788/qwen3.5-4b-arabic-grpo-v2)..."
python "$EVAL_SCRIPT_PATH" \
  --model unsloth/Qwen3.5-4B \
  --adapter-path aziz9788/qwen3.5-4b-arabic-grpo-v2 \
  --output-dir ./outputs/official_eval_grpo_v2

echo ""
echo "================================================================="
echo "OFFICIAL VAST.AI EVALUATION COMPLETE! 🏆"
echo "Base Model Summary : ./outputs/official_eval_base_model/summary.json"
echo "GRPO_V2 Summary    : ./outputs/official_eval_grpo_v2/summary.json"
echo "================================================================="
