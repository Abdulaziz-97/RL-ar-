@echo off
echo Starting SFT Distillation (Phase 1)...
call .venv\Scripts\activate.bat
set PYTHONPATH=src;rlvr
python -m rlvr_pipeline sft --config configs/qwen_4b_qlora.yaml --data data/arabic_reasoning_coldstart.jsonl --output runs/sft --num-train-epochs 2
pause
