#!/bin/bash
# GPU verification gates for V4 (run on Vast.ai after CPU pytest is green).
# Order: 1-GPU 2-step → 2-GPU 2-step DDP → 20-step canary + offline probe.
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/qwen_4b_2x5090_v4_sota.yaml}"
CANARY_DIR="${CANARY_DIR:-/workspace/outputs/v4_canary}"

echo "[gate] CPU static pytest subset"
python -m pytest "$REPO_ROOT/tests/test_v4_production_recipe.py" \
  "$REPO_ROOT/tests/test_v4_reward_contract.py" \
  "$REPO_ROOT/tests/test_v4_verification_gates.py" -q

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "No GPU detected; CPU gates only."
  exit 0
fi

echo "[gate] 1-GPU 2-step GRPO smoke"
rm -rf "$CANARY_DIR/smoke1"
CUDA_VISIBLE_DEVICES=0 python -m rlvr_pipeline.cli train \
  --config "$CONFIG" \
  --output "$CANARY_DIR/smoke1" \
  --max-steps 2 \
  --num-generations 2 \
  --per-device-batch-size 2 \
  --grad-accum 1

echo "[gate] adapter-only integrity"
python -m rlvr_pipeline.checkpoint_integrity --checkpoint "$CANARY_DIR/smoke1" --require-adapter-only || \
  python -m rlvr_pipeline.checkpoint_integrity --checkpoint "$(ls -d $CANARY_DIR/smoke1/checkpoint-* | sort -V | tail -n1)" --require-adapter-only

if [ "$NUM_GPUS" -ge 2 ]; then
  echo "[gate] 2-GPU 2-step DDP smoke"
  rm -rf "$CANARY_DIR/smoke2"
  torchrun --nproc_per_node=2 -m rlvr_pipeline.cli train \
    --config "$CONFIG" \
    --output "$CANARY_DIR/smoke2" \
    --max-steps 2 \
    --num-generations 2 \
    --per-device-batch-size 2 \
    --grad-accum 1
fi

echo "[gate] 20-step canary"
rm -rf "$CANARY_DIR/canary20"
if [ "$NUM_GPUS" -ge 2 ]; then
  torchrun --nproc_per_node=2 -m rlvr_pipeline.cli train \
    --config "$CONFIG" \
    --output "$CANARY_DIR/canary20" \
    --max-steps 20
else
  CUDA_VISIBLE_DEVICES=0 python -m rlvr_pipeline.cli train \
    --config "$CONFIG" \
    --output "$CANARY_DIR/canary20" \
    --max-steps 20
fi

CKPT=$(ls -d "$CANARY_DIR"/canary20/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "$CANARY_DIR/canary20")
python -m rlvr_pipeline.checkpoint_integrity --checkpoint "$CKPT" --require-adapter-only

echo "[gate] offline probe (optional; requires [eval] extra)"
if python -c "import vllm" 2>/dev/null; then
  python "$REPO_ROOT/scripts/run_benchmark_probe.py" --checkpoint "$CKPT" --base-model "Qwen/Qwen3.5-4B" || true
fi

echo "ALL GPU GATES COMPLETE"
