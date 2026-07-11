"""
SOTA Arabic Reasoning RLVR Pipeline.

Wraps TRL's GRPOTrainer (2026 SOTA) with:
  - Custom Arabic reward functions (correctness, format, language, anti-hack penalties)
  - QLoRA for 8GB VRAM (RTX 2080 Super)
  - Dr. GRPO loss (length-bias-free), batch-scale rewards, asymmetric DAPO clipping
  - GRPO / GSPO / GSPO-token switchable via config flag
  - Continuous batching for Windows-compatible fast generation
  - wandb monitoring

Plug-and-play: `python -m rlvr_sota train --config config.yaml --data data.jsonl`
"""

from rlvr_sota.config import SOTAConfig
from rlvr_sota.trainer import build_trainer

__all__ = ["SOTAConfig", "build_trainer"]
