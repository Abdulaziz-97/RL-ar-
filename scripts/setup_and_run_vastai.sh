#!/bin/bash
# Master Setup and Execution Script for Fresh Vast.ai GPU Instances (Multi-GPU & UV Accelerated)
set -e

echo "================================================================="
echo "INITIALIZING FRESH VAST.AI GPU INSTANCE FOR SAUDI-LLM GRPO V3"
echo "================================================================="

# 1. Environment Variables & Storage Redirection
export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf_cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export VLLM_USE_FLASHINFER_SAMPLER="0"
export VLLM_ENFORCE_EAGER="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export NCCL_IGNORE_DISABLED_P2P="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

mkdir -p /workspace/outputs
mkdir -p "$HF_HOME"
mkdir -p "$TMPDIR"

# Detect Available GPU Count
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=1; fi
echo "[System Diagnostic] Detected $NUM_GPUS active NVIDIA GPU(s)."

# 2. Fast Install via UV Package Manager
echo "================================================================="
echo "[1/4] Installing UV Package Manager & Accelerated Dependencies..."
echo "================================================================="
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
    echo "⚡ Using UV to install PyTorch, vLLM, TRL, PEFT, and RLVR pipeline..."
    uv pip install --system --break-system-packages -e /workspace/RL-ar-
    uv pip install --system --break-system-packages vllm lm-eval transformers peft accelerate datasets trl flash-attn --no-build-isolation || true
else
    echo "Installing via standard pip..."
    pip install -e /workspace/RL-ar- --no-deps
    pip install vllm lm-eval transformers peft accelerate datasets trl || true
fi

# 3. Data Curation Check
echo "================================================================="
echo "[2/4] Curating V3 Hard Dataset (Filtering pass@8=1.0 Trivial Items)..."
echo "================================================================="
python3 /workspace/RL-ar-/scripts/filter_hard_dataset.py

# 4. GRPO V3 Parallel Multi-GPU Training Launch
echo "================================================================="
echo "[3/4] LAUNCHING GRPO V3 PARALLEL MULTI-GPU TRAINING ($NUM_GPUS GPUs)"
echo "================================================================="

CONFIG_FILE="/workspace/RL-ar-/configs/qwen_4b_2x5090_v3_sota.yaml"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Running PyTorch Distributed Data Parallel (DDP) on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS -m rlvr_pipeline.cli train --config "$CONFIG_FILE"
else
    echo "🚀 Running Single GPU GRPO Training..."
    python3 -m rlvr_pipeline.cli train --config "$CONFIG_FILE"
fi

# 5. Parallel Generative Evaluation across 23,842 Test Questions
echo "================================================================="
echo "[4/4] STARTING PARALLEL GENERATIVE EVALUATION ($NUM_GPUS GPUs)"
echo "================================================================="

# Clean any lingering background vLLM worker processes
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f python3 2>/dev/null || true
rm -rf /dev/shm/vllm* /dev/shm/torch* /dev/shm/nccl* 2>/dev/null || true
sleep 2

GENERATIVE_SCRIPT="/workspace/RL-ar-/official_eval/run_araeval_generative.py"
OUTPUT_DIR="/workspace/outputs/generative_eval_grpo_v3"
TRAINED_CHECKPOINT=$(ls -d /workspace/RL-ar-/outputs/qwen_4b_2x5090_v3_run/checkpoint-* 2>/dev/null | tail -n 1 || echo "/workspace/RL-ar-/outputs/qwen_4b_2x5090_v3_run")

echo ">>> Evaluating Final GRPO_V3 Checkpoint: $TRAINED_CHECKPOINT..."
python3 "$GENERATIVE_SCRIPT" \
  --model unsloth/Qwen3.5-4B \
  --adapter-path "$TRAINED_CHECKPOINT" \
  --enable-thinking \
  --max-lora-rank 128 \
  --output-dir "$OUTPUT_DIR"

echo "================================================================="
echo "ALL STAGES COMPLETE SUCCESSFULLY! 🏆"
echo "GRPO V3 Model Checkpoint : $TRAINED_CHECKPOINT"
echo "Generative Evaluation    : $OUTPUT_DIR/summary.json"
echo "================================================================="
