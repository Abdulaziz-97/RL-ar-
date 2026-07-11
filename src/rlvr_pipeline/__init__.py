"""
Arabic Reasoning RLVR Pipeline.

Wraps TRL's GRPOTrainer (2026) with:
  - Custom Arabic reward functions (correctness, format, language, anti-hack penalties)
  - QLoRA for 8GB VRAM (RTX 2080 Super)
  - Dr. GRPO loss (length-bias-free), batch-scale rewards, asymmetric DAPO clipping
  - GRPO / GSPO / GSPO-token switchable via config flag
  - Continuous batching for Windows-compatible fast generation
  - wandb monitoring

Plug-and-play: `python -m rlvr_pipeline train --config config.yaml --data data.jsonl`
"""

from rlvr_pipeline.config import PipelineConfig
from rlvr_pipeline.trainer import build_trainer

__all__ = ["PipelineConfig", "build_trainer"]
