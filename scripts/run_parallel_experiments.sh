#!/bin/bash
# Parallel Dual-GPU Experiment Launcher
# GPU 0: Experiment A (LoRA V3a with Stage 1 SFT Chaining)
# GPU 1: Experiment B (Full-Parameter Fine-Tuning V3b Direct)

set -e

echo "================================================================="
echo "LAUNCHING DUAL-GPU PARALLEL EXPERIMENTS"
echo "================================================================="
echo "  * GPU 0: Experiment A — LoRA V3a SOTA Recipe"
echo "  * GPU 1: Experiment B — Full-Parameter FT V3b Recipe"
echo "================================================================="

# Storage & Environment Setup
export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf_cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

mkdir -p /workspace/outputs/exp_a_v3a /workspace/outputs/exp_b_v3b

# Curate V3 Hard Dataset
python3 /workspace/RL-ar-/scripts/filter_hard_dataset.py

# 1. Launch Experiment A on GPU 0 (Background Process)
echo "🚀 Starting Experiment A on GPU 0 (LoRA V3a)..."
CUDA_VISIBLE_DEVICES=0 python3 -m rlvr_pipeline.cli train \
  --config /workspace/RL-ar-/configs/qwen_4b_2x5090_v3_sota.yaml \
  --output /workspace/outputs/exp_a_v3a > /workspace/outputs/exp_a.log 2>&1 &
PID_A=$!

# 2. Launch Experiment B on GPU 1 (Background Process)
echo "🚀 Starting Experiment B on GPU 1 (Full-FT V3b)..."
CUDA_VISIBLE_DEVICES=1 python3 -m rlvr_pipeline.cli train \
  --config /workspace/RL-ar-/configs/qwen_4b_full_ft_v3.yaml \
  --output /workspace/outputs/exp_b_v3b > /workspace/outputs/exp_b.log 2>&1 &
PID_B=$!

echo ""
echo "================================================================="
echo "BOTH EXPERIMENTS RUNNING IN PARALLEL!"
echo "  * Monitor GPU 0 (Exp A): tail -f /workspace/outputs/exp_a.log"
echo "  * Monitor GPU 1 (Exp B): tail -f /workspace/outputs/exp_b.log"
echo "================================================================="

# Wait for both background tasks to complete
wait $PID_A
echo "✅ Experiment A (GPU 0) Finished!"

wait $PID_B
echo "✅ Experiment B (GPU 1) Finished!"

echo "================================================================="
echo "DUAL-GPU PARALLEL RUNS COMPLETE!"
echo "================================================================="
