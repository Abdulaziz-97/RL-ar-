@echo off
echo Starting GRPO Training (Phase 2)...
call .venv\Scripts\activate.bat
set PYTHONPATH=src;rlvr
python -m rlvr_pipeline train --config configs/qwen_4b_qlora.yaml --data data/arabic_reasoning_rlvr.jsonl --output runs/grpo --sft-checkpoint runs/sft --max-steps 150
pause
