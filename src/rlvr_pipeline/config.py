"""
Training configuration.

Maps user-facing settings to TRL's GRPOConfig + QLoRA (bitsandbytes + peft).
Supports single-flag switching between GRPO / GSPO / GSPO-token via
`importance_sampling_level`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from trl import GRPOConfig


DEFAULT_SYSTEM_PROMPT = (
    "قواعد التنسيق - يجب الالتزام بها بالضبط:\n"
    "1. ابدأ بـ: <think>\n"
    "2. اكتب تفكيرك خطوة بخطوة\n"
    "3. أنهِ التفكير بـ: </think>\n"
    "4. ثم اكتب: <answer>\n"
    "5. ضع الإجابة النهائية فقط (رقم أو نص قصير)\n"
    "6. أنهِ بـ: </answer>\n"
    "مثال: <think>عملية التفكير</think><answer>42</answer>\n"
    "لا يُقبل أي تنسيق آخر."
)

LossType = Literal["grpo", "dapo", "dr_grpo", "sapo", "bnpo"]
ImportanceSamplingLevel = Literal["token", "sequence", "sequence_token"]
ScaleRewards = Literal["group", "batch", "off"]


@dataclass
class PipelineConfig:
    # ── Model ──
    model_name: str = "Qwen/Qwen3.5-2B"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ── LoRA / QLoRA ──
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "out_proj", "in_proj_z",
            "in_proj_b", "in_proj_a",
        ]
    )

    # ── Data ──
    train_data_path: str = ""
    eval_data_path: str = ""
    coldstart_data_path: str = ""
    sft_checkpoint_path: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # ── Generation (GRPO group sampling) ──
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20

    # ── Training ──
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    seed: int = 42

    # ── Algorithm (2026) ──
    loss_type: LossType = "dr_grpo"
    scale_rewards: ScaleRewards = "batch"
    beta: float = 0.04
    epsilon: float = 0.2
    epsilon_high: Optional[float] = 0.28
    importance_sampling_level: ImportanceSamplingLevel = "token"
    mask_truncated_completions: bool = True

    # ── Reward weights (0.6 correct + 0.2 format + 0.05 lang − 0.5 leak − 0.3 struct) ──
    reward_weights: list[float] = field(
        default_factory=lambda: [0.6, 0.2, 0.05, 0.5, 0.3]
    )

    # ── Failure Mining (RL-ZVP / POPO / discard) ──
    zero_variance_strategy: Literal["direct_scoring", "replay_buffer", "discard"] = "direct_scoring"
    replay_buffer_size: int = 512

    # ── Curriculum Learning (E2H Reasoner Gaussian schedule) ──
    curriculum_schedule_type: Literal["gaussian", "fixed_switch", "random_mix", "none"] = "gaussian"
    sigma_fraction: float = 0.2

    # ── CRPS (progressive suffix hints for hard stage) ──
    enable_crps: bool = True
    crps_max_age_steps: int = 100
    crps_max_traces: int = 256

    # ── Infrastructure ──
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_transformers_continuous_batching: bool = True
    report_to: str = "wandb"
    use_wandb: bool = False
    wandb_project: str = "arabic-reasoning-rlvr"

    # ── Logging ──
    log_completions: bool = True
    num_completions_to_print: int = 4

    # ── Extra ──
    stop_strings: Optional[list[str]] = None
    max_steps: Optional[int] = None

    # ──────────────────────────────────────────────
    #  Serialization
    # ──────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ──────────────────────────────────────────────
    #  Builders
    # ──────────────────────────────────────────────

    def build_grpo_config(self, include_model_init: bool = True) -> GRPOConfig:
        scale_rewards_val: Any
        if self.scale_rewards == "off":
            scale_rewards_val = False
        else:
            scale_rewards_val = self.scale_rewards

        generation_kwargs: dict[str, Any] = {"top_p": self.top_p, "top_k": self.top_k}
        if self.stop_strings:
            generation_kwargs["stop_strings"] = self.stop_strings

        kwargs: dict[str, Any] = dict(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            optim=self.optim,
            seed=self.seed,
            num_generations=self.num_generations,
            max_completion_length=self.max_completion_length,
            temperature=self.temperature,
            generation_kwargs=generation_kwargs,
            loss_type=self.loss_type,
            scale_rewards=scale_rewards_val,
            beta=self.beta,
            epsilon=self.epsilon,
            epsilon_high=self.epsilon_high,
            importance_sampling_level=self.importance_sampling_level,
            mask_truncated_completions=self.mask_truncated_completions,
            reward_weights=self.reward_weights,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            use_transformers_continuous_batching=self.use_transformers_continuous_batching,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            report_to=self.report_to if self.use_wandb else "none",
            log_completions=self.log_completions,
            num_completions_to_print=self.num_completions_to_print,
            use_vllm=False,
        )

        if include_model_init:
            kwargs["model_init_kwargs"] = self.build_model_init_kwargs()

        if self.eval_steps is not None:
            kwargs["eval_steps"] = self.eval_steps
            kwargs["eval_strategy"] = "steps"

        if self.max_steps is not None:
            kwargs["max_steps"] = self.max_steps

        return GRPOConfig(**kwargs)

    def build_quantization_config(self):
        if not self.load_in_4bit:
            return None
        from transformers import BitsAndBytesConfig
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map[self.bnb_4bit_compute_dtype],
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    def build_peft_config(self):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

    def build_model_init_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        quant_config = self.build_quantization_config()
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        if self.bf16:
            kwargs["torch_dtype"] = "bfloat16"
        return kwargs
