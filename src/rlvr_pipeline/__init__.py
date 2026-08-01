"""
Arabic Reasoning RLVR Pipeline.

Wraps TRL's GRPOTrainer with Arabic rewards, QLoRA, and curriculum options.

Compatibility shims are version-gated and applied only when needed. Training
does not import or initialize vLLM at package import time.
"""

from __future__ import annotations

import importlib.metadata as _md


def _pkg_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _apply_transformers_hub_cache_shim() -> None:
    """transformers>=5.14 renamed TRANSFORMERS_CACHE; keep older callers working."""
    try:
        import transformers.utils.hub as _hub

        if not hasattr(_hub, "TRANSFORMERS_CACHE"):
            _hub.TRANSFORMERS_CACHE = getattr(_hub, "HF_HUB_CACHE", None)
    except Exception:
        pass


def _apply_qwen35_use_cache_shim() -> None:
    """Absorb unexpected use_cache kwarg during Qwen3.5 construction under TRL."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        if getattr(Qwen3_5ForCausalLM.__init__, "_rlvr_patched", False):
            return
        _orig = Qwen3_5ForCausalLM.__init__

        def _patched(self, config, *args, use_cache=None, **kwargs):
            return _orig(self, config, *args, **kwargs)

        _patched._rlvr_patched = True  # type: ignore[attr-defined]
        Qwen3_5ForCausalLM.__init__ = _patched  # type: ignore[method-assign]
    except Exception:
        pass


def _apply_pretrained_warnings_shim() -> None:
    try:
        from transformers import PreTrainedModel

        if not hasattr(PreTrainedModel, "warnings_issued"):
            PreTrainedModel.warnings_issued = {}
    except Exception:
        pass


def apply_training_shims() -> None:
    """Apply version-checked training compatibility shims (no vLLM)."""
    _apply_transformers_hub_cache_shim()
    _apply_qwen35_use_cache_shim()
    _apply_pretrained_warnings_shim()


# Apply lightweight shims on import; never import vLLM here.
apply_training_shims()
