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
  --enable-thinking \
  --limit 50

# Safely upload checkpoint adapters, merged model, and evaluation logs to Hugging Face Hub
export HF_TOKEN="${HF_TOKEN:-$(python3 -c 'from scripts.upload_artifacts import get_hf_token; print(get_hf_token())')}"
python3 scripts/upload_artifacts.py \
  --hf-repo "aziz9788/qwen3.5-4b-arabic-grpo-v4-checkpoint-200-adapter"
