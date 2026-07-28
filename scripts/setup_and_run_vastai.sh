#!/bin/bash
# Master Setup and Execution Script for Fresh Vast.ai GPU Instances
set -e

echo "================================================================="
echo "INITIALIZING FRESH VAST.AI GPU INSTANCE FOR SAUDI-LLM EVALUATION"
echo "================================================================="

# 1. Environment Variables (ensure HF_TOKEN is exported in shell)
export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
export VLLM_USE_FLASHINFER_SAMPLER="0"
export VLLM_ENFORCE_EAGER="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

mkdir -p /workspace/outputs
mkdir -p "$HF_HOME"

# 2. Install uv package manager & Python dependencies
echo "[1/4] Installing uv package manager & Python dependencies..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
    echo "Using uv to install vllm, lm-eval, peft, accelerate, datasets..."
    uv pip install --system --break-system-packages vllm lm-eval transformers peft accelerate datasets trl
else
    python3 -m pip install -q --break-system-packages vllm lm-eval transformers peft accelerate datasets trl || true
fi

# 3. Locate Internal Official Evaluation Script
EVAL_SCRIPT="/workspace/RL-ar-/official_eval/run_araeval.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    EVAL_SCRIPT="./official_eval/run_araeval.py"
fi

echo "[3/4] Script located at: $EVAL_SCRIPT"

# 4. Run Official Benchmark Suite
echo "================================================================="
echo "[4/4] STARTING OFFICIAL BENCHMARKS FOR BASE & GRPO_V2 MODELS"
echo "================================================================="

# Clean any lingering background vLLM worker processes, stale IPC sockets & corrupted CUDA driver handles
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f python3 2>/dev/null || true
pkill -9 -f python 2>/dev/null || true
fuser -k -9 /dev/nvidia* 2>/dev/null || true
rm -rf /dev/shm/vllm* /dev/shm/torch* /dev/shm/nccl* 2>/dev/null || true
sleep 2

echo ""
echo ">>> STEP A: Evaluating Base Model (unsloth/Qwen3.5-4B)..."
python3 "$EVAL_SCRIPT" \
  --model unsloth/Qwen3.5-4B \
  --max-batch-size 8 \
  --output-dir /workspace/outputs/official_eval_base_model

echo ""
echo ">>> STEP B: Evaluating GRPO_V2 Model (aziz9788/qwen3.5-4b-arabic-grpo-v2)..."
python3 "$EVAL_SCRIPT" \
  --model unsloth/Qwen3.5-4B \
  --adapter-path aziz9788/qwen3.5-4b-arabic-grpo-v2 \
  --max-batch-size 8 \
  --output-dir /workspace/outputs/official_eval_grpo_v2

echo ""
echo "================================================================="
echo "OFFICIAL VAST.AI EVALUATION COMPLETE! 🏆"
echo "Base Model Summary : /workspace/outputs/official_eval_base_model/summary.json"
echo "GRPO_V2 Summary    : /workspace/outputs/official_eval_grpo_v2/summary.json"
echo "================================================================="
