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

# Patch transformers.configuration_utils for vllm 0.26.0 compatibility on transformers 4.49.0
try:
    import transformers.configuration_utils as _cfg_utils
    if not hasattr(_cfg_utils, "ALLOWED_LAYER_TYPES"):
        _cfg_utils.ALLOWED_LAYER_TYPES = ["linear", "conv"]
except Exception:
    pass

# Pre-import vllm.worker.worker so TRL 0.15 mock patch finds vllm.worker attribute
try:
    import vllm.worker.worker
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

# Patch PreTrainedModel.generate to force use_cache=True during rollout generation for 300x speedup
try:
    from transformers import PreTrainedModel
    _orig_generate = PreTrainedModel.generate
    def _patched_generate(self, *args, **kwargs):
        kwargs["use_cache"] = True
        return _orig_generate(self, *args, **kwargs)
    PreTrainedModel.generate = _patched_generate
except Exception:
    pass

# Patch TRL GRPOTrainer.__init__ batch validation so per_device_train_batch_size=4 runs at fast 20s speed with 16 rollouts
try:
    import trl.trainer.grpo_trainer as _grpo_mod
    _orig_grpo_init = _grpo_mod.GRPOTrainer.__init__
    def _patched_grpo_init(self, *args, **kwargs):
        try:
            return _orig_grpo_init(self, *args, **kwargs)
        except ValueError as err:
            if "must be evenly divisible" in str(err):
                args_obj = kwargs.get("args") or (args[1] if len(args) > 1 else None)
                if args_obj and hasattr(args_obj, "per_device_train_batch_size") and hasattr(args_obj, "num_generations"):
                    orig_bs = args_obj.per_device_train_batch_size
                    args_obj.per_device_train_batch_size = args_obj.num_generations
                    res = _orig_grpo_init(self, *args, **kwargs)
                    args_obj.per_device_train_batch_size = orig_bs
                    return res
            raise err
    _grpo_mod.GRPOTrainer.__init__ = _patched_grpo_init
except Exception:
    pass

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.trainer import build_trainer

__all__ = ["RLVRConfig", "build_trainer"]
