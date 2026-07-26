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

# Backward compatibility patch for transformers hub cache in transformers 5.14+
try:
    import transformers.utils.hub as _hub
    if not hasattr(_hub, "TRANSFORMERS_CACHE"):
        _hub.TRANSFORMERS_CACHE = getattr(_hub, "HF_HUB_CACHE", None)
except Exception:
    pass

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

# Patch TRL GRPOTrainer.__init__ batch validation so per_device_train_batch_size=4
# produces 4 prompts/step (not 1). TRL requires per_device_train_batch_size to be
# divisible by num_generations during validation, so we temporarily set it to
# num_generations, let init succeed, then fix _train_batch_size back to the real
# intended value (e.g. 4) so get_train_dataloader fetches 4 prompts per step.
try:
    import trl.trainer.grpo_trainer as _grpo_mod
    _orig_grpo_init = _grpo_mod.GRPOTrainer.__init__
    def _patched_grpo_init(self, *args, **kwargs):
        args_obj = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        intended_bs = getattr(args_obj, "per_device_train_batch_size", None) if args_obj else None
        try:
            _orig_grpo_init(self, *args, **kwargs)
        except ValueError as err:
            if "must be evenly divisible" in str(err) and args_obj is not None and intended_bs is not None:
                # TRL validation rejects batch_size not divisible by num_generations.
                # Temporarily set to num_generations to pass validation, then fix after.
                args_obj.per_device_train_batch_size = args_obj.num_generations
                _orig_grpo_init(self, *args, **kwargs)
                args_obj.per_device_train_batch_size = intended_bs
            else:
                raise err
        # Restore _train_batch_size to actual intended prompts-per-step so that
        # get_train_dataloader fetches `intended_bs` prompts, not num_generations.
        if intended_bs is not None and hasattr(self, "_train_batch_size"):
            self._train_batch_size = intended_bs
    _grpo_mod.GRPOTrainer.__init__ = _patched_grpo_init
except Exception:
    pass

# Patch TRL entropy_from_logits to process in 256-token chunks, avoiding giant 11.37 GB memory spikes
try:
    import torch
    import trl.trainer.utils as _trl_utils
    import trl.trainer.grpo_trainer as _grpo_mod

    def _chunked_entropy_from_logits(logits: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        shape_except_last = logits.shape[:-1]
        num_classes = logits.shape[-1]
        flat_logits = logits.reshape(-1, num_classes)
        num_tokens = flat_logits.shape[0]

        entropies = []
        for i in range(0, num_tokens, chunk_size):
            chunk = flat_logits[i : i + chunk_size]
            logps = chunk.log_softmax(dim=-1)
            chunk_entropy = -(torch.exp(logps) * logps).sum(dim=-1)
            entropies.append(chunk_entropy)

        return torch.cat(entropies, dim=0).reshape(shape_except_last)

    _trl_utils.entropy_from_logits = _chunked_entropy_from_logits
    if hasattr(_grpo_mod, "entropy_from_logits"):
        _grpo_mod.entropy_from_logits = _chunked_entropy_from_logits
except Exception:
    pass

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.trainer import build_trainer

__all__ = ["RLVRConfig", "build_trainer"]
