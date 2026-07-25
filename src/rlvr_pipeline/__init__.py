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

# Backward compatibility patch for transformers hub cache
try:
    import transformers.utils.hub as _hub
    if not hasattr(_hub, "TRANSFORMERS_CACHE"):
        _hub.TRANSFORMERS_CACHE = getattr(_hub, "HF_HUB_CACHE", None)
except Exception:
    pass

# Patch Qwen3_5ForCausalLM.__init__ to safely absorb use_cache kwarg passed by TRL
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    _orig_qwen_init = Qwen3_5ForCausalLM.__init__
    def _patched_qwen_init(self, config, *args, use_cache=None, **kwargs):
        _orig_qwen_init(self, config, *args, **kwargs)
    Qwen3_5ForCausalLM.__init__ = _patched_qwen_init
except Exception:
    pass

# Patch PreTrainedModel.warnings_issued for TRL 0.15+ estimate_tokens compatibility
try:
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "warnings_issued"):
        PreTrainedModel.warnings_issued = {}
except Exception:
    pass

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.trainer import build_trainer

__all__ = ["RLVRConfig", "build_trainer"]
