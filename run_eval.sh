cd /workspace/RL-ar-
export PYTHONPATH="/workspace/RL-ar-/src:$PYTHONPATH"

# Evaluate the trained GRPO adapter on the T06 instruction lineage (not raw base).
CUDA_VISIBLE_DEVICES=0 python3 official_eval/run_araeval_generative.py \
  --model "Qwen/Qwen3.5-4B" \
  --instruction-adapter "aziz9788/T06__qwen35-mixed-v6-lr1e5" \
  --adapter-path "/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/checkpoint-200" \
  --output-dir "/workspace/outputs/generative_eval_checkpoint_200" \
  --tasks araeval_aramath araeval_ifeval araeval_arapro \
  --enable-thinking \
  --engine hf \
  --limit 50

# Safely upload checkpoint adapters and evaluation logs to Hugging Face Hub
export HF_TOKEN="${HF_TOKEN:-$(python3 -c 'from scripts.upload_artifacts import get_hf_token; print(get_hf_token())')}"
python3 scripts/upload_artifacts.py \
  --hf-repo "aziz9788/qwen3.5-4b-arabic-grpo-v4-checkpoint-200-adapter"
