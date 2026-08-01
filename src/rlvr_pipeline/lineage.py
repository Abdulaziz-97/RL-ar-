"""Shared model lineage loader for SFT, GRPO, probes, and evaluation.

Canonical V4 lineage:
  Qwen/Qwen3.5-4B base
    → strict T06 instruction-adapter merge
    → fresh SFT LoRA (adapter-only)
    → strict trainable GRPO adapter load
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rlvr_pipeline.trainer import fix_chat_template, load_sft_adapter_strict


LINEAGE_FILENAME = "lineage.json"


@dataclass
class LineageMeta:
    base_model: str
    instruction_adapter: Optional[str] = None
    sft_parent: Optional[str] = None
    adapter_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    stage: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageMeta":
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


def file_sha256(path: str | Path, max_bytes: int = 8_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def write_lineage(checkpoint_dir: str | Path, meta: LineageMeta) -> Path:
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / LINEAGE_FILENAME
    path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_lineage(checkpoint_dir: str | Path) -> Optional[LineageMeta]:
    path = Path(checkpoint_dir) / LINEAGE_FILENAME
    if not path.is_file():
        return None
    return LineageMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_lineage_model(
    *,
    base_model: str,
    instruction_adapter: Optional[str] = None,
    trainable_adapter: Optional[str] = None,
    merge_instruction: bool = True,
    is_trainable: bool = False,
    torch_dtype=None,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
):
    """Load base → optional instruction merge → optional trainable adapter (strict)."""
    import torch
    from peft import PeftModel, get_peft_model, LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(model.config, "text_config"):
        model.config.text_config = model.config

    if instruction_adapter and merge_instruction:
        print(f"[lineage] Merging instruction adapter: {instruction_adapter}", flush=True)
        peft = PeftModel.from_pretrained(model, instruction_adapter)
        model = peft.merge_and_unload()
        del peft

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if trainable_adapter:
        adapter_cfg_path = Path(trainable_adapter) / "adapter_config.json"
        if not adapter_cfg_path.is_file():
            raise FileNotFoundError(f"Missing adapter_config.json in {trainable_adapter}")
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        targets = adapter_cfg.get("target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        peft_config = LoraConfig(
            r=int(adapter_cfg.get("r", 64)),
            lora_alpha=int(adapter_cfg.get("lora_alpha", 128)),
            lora_dropout=float(adapter_cfg.get("lora_dropout", 0.05)),
            target_modules=list(targets),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        load_sft_adapter_strict(model, trainable_adapter)
        if is_trainable:
            model.train()
        else:
            model.eval()

    fix_chat_template(tokenizer)
    return model, tokenizer
