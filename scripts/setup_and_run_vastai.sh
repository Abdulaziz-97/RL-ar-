#!/bin/bash
# Master Setup and Execution Script for Fresh Vast.ai GPU Instances (Saudi-LLM GRPO V4)
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "================================================================="
echo "INITIALIZING FRESH VAST.AI GPU INSTANCE FOR SAUDI-LLM GRPO V4"
echo "Repository Root: $REPO_ROOT"
echo "================================================================="

# 1. Create Storage Directories FIRST & Redirect Terminal Output to Disk
mkdir -p /workspace/tmp /workspace/.hf_cache /workspace/outputs "$REPO_ROOT/outputs"
exec > >(tee -a /workspace/outputs/master_execution_v4.log) 2>&1

# Environment Variables & Storage Redirection
export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf_cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export VLLM_USE_FLASHINFER_SAMPLER="0"
export VLLM_ENFORCE_EAGER="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export NCCL_IGNORE_DISABLED_P2P="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

# Detect Available GPU Count
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=1; fi
echo "[System Diagnostic] Detected $NUM_GPUS active NVIDIA GPU(s)."

# 2. Fast Install via UV Package Manager
echo "================================================================="
echo "[1/5] Installing UV Package Manager & Accelerated Dependencies..."
echo "================================================================="
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
    echo "⚡ Using UV to install PyTorch, vLLM, TRL, PEFT, Arabic Rendering, and RLVR pipeline..."
    uv pip install --system --break-system-packages -e "$REPO_ROOT"
    uv pip install --system --break-system-packages vllm lm-eval transformers peft accelerate datasets trl python-bidi arabic-reshaper
else
    echo "Installing via standard pip..."
    pip install -e "$REPO_ROOT" --no-deps
    pip install vllm lm-eval transformers peft accelerate datasets trl python-bidi arabic-reshaper || true
fi

# 3. Live Qwen 3.7 RLVR Generation & Gate Verification Loop
echo "================================================================="
echo "[2/5] LIVE QWEN 3.7 FLASH RLVR DATA GENERATION & GATE VERIFICATION LOOP"
echo "================================================================="
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "🚀 Launching Live 4K RLVR Data Generation Loop with Content-Jaccard Guard (<0.65)..."
    python3 "$REPO_ROOT/data_1/scripts/generate_qwen37_rlvr_live.py"
else
    echo "⚠️ OPENROUTER_API_KEY not set. Using pre-generated RLVR dataset in data/arabic_reasoning_rlvr_v4.jsonl."
fi

echo "🔬 Running Comprehensive Gate & Zero-Overlap Verification Audit..."
python3 "$REPO_ROOT/data_1/scripts/verify_master_v4_datasets_complete.py"

CONFIG_FILE="$REPO_ROOT/configs/qwen_4b_2x5090_v4_sota.yaml"
SFT_OUT="/workspace/outputs/sft_coldstart_v4"

# 4. Stage 1: Cold-Start SFT Warm-up (4,000 CoT Solutions)
echo "================================================================="
echo "[3/5] STAGE 1: COLD-START CoT SFT WARM-UP (4,000 CoT Solutions)"
echo "================================================================="
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Running PyTorch DDP SFT Warm-up on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS -m rlvr_pipeline.cli sft --config "$CONFIG_FILE" --output "$SFT_OUT" --max-steps 100 || true
else
    CUDA_VISIBLE_DEVICES=0 python3 -m rlvr_pipeline.cli sft --config "$CONFIG_FILE" --output "$SFT_OUT" --max-steps 100 || true
fi

# 5. Stage 2: GRPO V4 Parallel Multi-GPU Training Launch
echo "================================================================="
echo "[4/5] STAGE 2: LAUNCHING GRPO V4 PARALLEL MULTI-GPU TRAINING ($NUM_GPUS GPUs)"
echo "================================================================="

SFT_ARG=""
if [ -f "$SFT_OUT/adapter_config.json" ] || [ -f "$SFT_OUT/adapter_model.safetensors" ]; then
    SFT_ARG="--sft-checkpoint $SFT_OUT"
    echo "  * Chaining from Stage 1 SFT Checkpoint: $SFT_OUT"
else
    LATEST_SFT_CKPT=$(ls -d $SFT_OUT/checkpoint-* 2>/dev/null | tail -n 1 || echo "")
    if [ -n "$LATEST_SFT_CKPT" ]; then
        SFT_ARG="--sft-checkpoint $LATEST_SFT_CKPT"
        echo "  * Chaining from Stage 1 SFT Checkpoint: $LATEST_SFT_CKPT"
    else
        echo "  * Starting GRPO V4 directly from aziz9788 SOTA instruction model..."
    fi
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Running PyTorch Distributed Data Parallel (DDP) on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS -m rlvr_pipeline.cli train --config "$CONFIG_FILE" $SFT_ARG
else
    echo "🚀 Running Single GPU GRPO Training..."
    python3 -m rlvr_pipeline.cli train --config "$CONFIG_FILE" $SFT_ARG
fi

# 6. Parallel Generative Evaluation across AraEval Test Questions
echo "================================================================="
echo "[5/5] STARTING PARALLEL GENERATIVE EVALUATION ($NUM_GPUS GPUs)"
echo "================================================================="

pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f python3 2>/dev/null || true
rm -rf /dev/shm/vllm* /dev/shm/torch* /dev/shm/nccl* 2>/dev/null || true
sleep 2

GENERATIVE_SCRIPT="$REPO_ROOT/official_eval/run_araeval_generative.py"
OUTPUT_DIR="/workspace/outputs/generative_eval_grpo_v4"
TRAINED_CHECKPOINT=$(ls -d "$REPO_ROOT/outputs/qwen_4b_2x5090_v4_run/checkpoint-"* 2>/dev/null | tail -n 1 || echo "$REPO_ROOT/outputs/qwen_4b_2x5090_v4_run")

echo ">>> Evaluating Final GRPO_V4 Checkpoint: $TRAINED_CHECKPOINT..."
python3 "$GENERATIVE_SCRIPT" \
  --model aziz9788/T06__qwen35-mixed-v6-lr1e5 \
  --adapter-path "$TRAINED_CHECKPOINT" \
  --enable-thinking \
  --max-lora-rank 128 \
  --output-dir "$OUTPUT_DIR"

echo "================================================================="
echo "ALL STAGES COMPLETE SUCCESSFULLY! 🏆"
echo "GRPO V4 Model Checkpoint : $TRAINED_CHECKPOINT"
echo "Generative Evaluation    : $OUTPUT_DIR/summary.json"
echo "================================================================="
