#!/bin/bash
# Master Setup and Execution Script for Fresh Vast.ai GPU Instances
set -e

echo "================================================================="
echo "INITIALIZING FRESH VAST.AI GPU INSTANCE FOR SAUDI-LLM EVALUATION"
echo "================================================================="

# 1. Environment Variables (ensure HF_TOKEN is exported in shell)
export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
export VLLM_USE_FLASHINFER_SAMPLER="0"
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

mkdir -p /workspace/outputs
mkdir -p "$HF_HOME"

# 2. System Packages & Python Dependencies Setup
echo "[1/4] Installing required system packages & Python libraries..."
apt-get update -qq && apt-get install -y -qq git git-lfs > /dev/null 2>&1 || true

python3 -m pip install --upgrade pip -q
python3 -m pip install -q vllm lm-eval transformers peft accelerate datasets trl || true

# 3. Verify / Clone Repositories
echo "[2/4] Verifying repository setups..."
if [ ! -d "/workspace/RL-ar-" ]; then
    echo "Cloning RL-ar- repository..."
    git clone -b Efficient-Arabic-Reasnoning-Pipeline https://github.com/Abdulaziz-97/RL-ar-.git /workspace/RL-ar-
fi

if [ ! -d "/workspace/Saudi-LLM" ]; then
    echo "Cloning Saudi-LLM repository..."
    git clone https://github.com/Abdulaziz-97/Saudi-LLM.git /workspace/Saudi-LLM || true
fi

# Locate run_araeval.py script
EVAL_SCRIPT="/workspace/Saudi-LLM/Eval/scripts/run_araeval.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    EVAL_SCRIPT="/workspace/RL-ar-/Saudi-LLM/Eval/scripts/run_araeval.py"
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: Official evaluation script run_araeval.py not found!"
    echo "Please ensure Saudi-LLM repo is placed at /workspace/Saudi-LLM"
    exit 1
fi

echo "[3/4] Script located at: $EVAL_SCRIPT"

# 4. Run Official Benchmark Suite
echo "================================================================="
echo "[4/4] STARTING OFFICIAL BENCHMARKS FOR BASE & GRPO_V2 MODELS"
echo "================================================================="

echo ""
echo ">>> STEP A: Evaluating Base Model (unsloth/Qwen3.5-4B)..."
python3 "$EVAL_SCRIPT" \
  --model unsloth/Qwen3.5-4B \
  --output-dir /workspace/outputs/official_eval_base_model

echo ""
echo ">>> STEP B: Evaluating GRPO_V2 Model (aziz9788/qwen3.5-4b-arabic-grpo-v2)..."
python3 "$EVAL_SCRIPT" \
  --model unsloth/Qwen3.5-4B \
  --adapter-path aziz9788/qwen3.5-4b-arabic-grpo-v2 \
  --output-dir /workspace/outputs/official_eval_grpo_v2

echo ""
echo "================================================================="
echo "OFFICIAL VAST.AI EVALUATION COMPLETE! 🏆"
echo "Base Model Summary : /workspace/outputs/official_eval_base_model/summary.json"
echo "GRPO_V2 Summary    : /workspace/outputs/official_eval_grpo_v2/summary.json"
echo "================================================================="
