fuser -k -9 /dev/nvidia* 2>/dev/null || true
pkill -9 -f python 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 2

# Cleanup stale merge folders and temporary caches to free disk space
rm -rf /workspace/RL-ar-/outputs/*/merged_eval_* 2>/dev/null || true
rm -rf /root/.cache/huggingface/hub/tmp* 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true

cd /workspace/RL-ar-
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 official_eval/run_araeval_generative.py \
  --model "Qwen/Qwen3.5-4B" \
  --adapter-path "/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/checkpoint-200" \
  --output-dir "/workspace/outputs/generative_eval_checkpoint_200" \
  --tasks araeval_aramath araeval_ifeval araeval_arapro \
  --limit 50
