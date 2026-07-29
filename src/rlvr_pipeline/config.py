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
class RLVRConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-2B"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA / QLoRA
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

    # Data
    train_data_path: str = ""
    eval_data_path: str = ""
    coldstart_data_path: str = ""
    sft_checkpoint_path: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Generation (GRPO group sampling)
    num_generations: int = 8
    max_completion_length: int = 384
    temperature: float = 1.15
    top_p: float = 0.95
    top_k: int = 20
    stop_strings: Optional[list[str]] = field(default_factory=lambda: ["</answer>"])

    # Training
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    seed: int = 42

    # Algorithm
    loss_type: LossType = "dr_grpo"
    scale_rewards: ScaleRewards = "batch"
    beta: float = 0.04
    epsilon: float = 0.2
    epsilon_high: Optional[float] = 0.28
    importance_sampling_level: ImportanceSamplingLevel = "token"
    mask_truncated_completions: bool = True

    # Entropy regularization (prevents entropy collapse — TRL native)
    # entropy_coef: coefficient for entropy bonus in the loss function
    # use_adaptive_entropy: dynamically adjusts entropy_coef based on entropy_target
    # entropy_target: target per-token entropy (nats); coefficient increases when below
    # entropy_coef_delta: step size for adaptive entropy coefficient adjustments
    entropy_coef: float = 0.0
    use_adaptive_entropy: bool = False
    entropy_target: float = 2.0
    entropy_coef_delta: float = 0.005

    # Reward weights: [correctness, format, language, answer_leak, structural_leak, length]
    reward_weights: list[float] = field(
        default_factory=lambda: [0.6, 0.2, 0.05, 0.5, 0.3, 0.15]
    )
    # GDPO: Normalize each reward independently before weighted sum (TRL native)
    # (NVlabs arXiv:2601.05242 — fixes reward advantage collapse with multi-reward + small groups)
    gdpo_decoupled_normalization: bool = False

    # Failure mining
    zero_variance_strategy: Literal["direct_scoring", "discard"] = "direct_scoring"

    # Curriculum
    curriculum_schedule_type: Literal["gaussian", "fixed_switch", "random_mix", "none"] = "gaussian"
    sigma_fraction: float = 0.2

    # CRPS
    enable_crps: bool = True
    crps_max_age_steps: int = 100
    crps_max_traces: int = 256

    # Stability guards
    enable_stability_callback: bool = True
    stability_consecutive: int = 2
    entropy_collapse_action: Literal["warn", "reduce_lr", "stop"] = "reduce_lr"
    entropy_lr_reduction_factor: float = 0.5
    enable_adaptive_beta: bool = True
    kl_near_zero_threshold: float = 1e-3
    stuck_beta: float = 0.02
    enable_adaptive_temperature: bool = False
    entropy_target_min: float = 2.0
    temp_bump: float = 0.15
    temp_max: float = 1.6
    temp_bump_cooldown_steps: int = 3
    early_diag_steps: int = 30

    # Infrastructure
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    use_transformers_continuous_batching: bool = False
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.40
    vllm_max_model_len: int = 4096
    attn_implementation: str = "flash_attention_2"
    ddp_find_unused_parameters: bool = False
    report_to: str = "wandb"
    use_wandb: bool = False
    wandb_project: str = "arabic-reasoning-rlvr"

    # Logging
    log_completions: bool = True
    num_completions_to_print: int = 4

    sft_learning_rate: float = 2.0e-4
    max_steps: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLVRConfig":
        import dataclasses
        config_path = Path(path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        base_dir = (
            config_path.parent.parent
            if config_path.parent.name.lower() == "configs"
            else config_path.parent
        )
        for key in ("train_data_path", "eval_data_path", "coldstart_data_path"):
            value = data.get(key)
            if value and not Path(value).is_absolute():
                data[key] = str((base_dir / value).resolve())

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def build_grpo_config(self, include_model_init: bool = True) -> GRPOConfig:
        scale_rewards_val: Any
        if self.scale_rewards == "off":
            scale_rewards_val = False
        else:
            scale_rewards_val = self.scale_rewards

        generation_kwargs: dict[str, Any] = {"top_p": self.top_p, "top_k": self.top_k}
        # stop_strings are attached via StopStringCriteria in build_trainer (TRL generate path).

        num_generations_eval = self.num_generations
        eval_batch_size = self.per_device_eval_batch_size or num_generations_eval

        kwargs: dict[str, Any] = dict(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            optim=self.optim,
            seed=self.seed,
            num_generations=self.num_generations,
            num_generations_eval=num_generations_eval,
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
            multi_objective_aggregation="normalize_then_sum" if self.gdpo_decoupled_normalization else "sum_then_normalize",
            # Entropy regularization (prevents entropy collapse)
            entropy_coef=self.entropy_coef,
            use_adaptive_entropy=self.use_adaptive_entropy,
            entropy_target=self.entropy_target,
            entropy_coef_delta=self.entropy_coef_delta,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            fp16=self.fp16,
            use_transformers_continuous_batching=self.use_transformers_continuous_batching,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            report_to=self.report_to if self.use_wandb else "none",
            log_completions=self.log_completions,
            num_completions_to_print=self.num_completions_to_print,
            use_vllm=getattr(self, "use_vllm", False),
            vllm_gpu_memory_utilization=getattr(self, "vllm_gpu_memory_utilization", 0.40),
            vllm_max_model_len=getattr(self, "vllm_max_model_len", 4096),
            ddp_find_unused_parameters=getattr(self, "ddp_find_unused_parameters", False),
        )

        if include_model_init:
            kwargs["model_init_kwargs"] = self.build_model_init_kwargs()

        if self.eval_steps is not None:
            kwargs["eval_steps"] = self.eval_steps
            kwargs["eval_strategy"] = "steps"

        if self.max_steps is not None:
            kwargs["max_steps"] = self.max_steps

        # Safely filter kwargs against GRPOConfig signature for cross-version compatibility
        import inspect
        sig = inspect.signature(GRPOConfig.__init__)
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return GRPOConfig(**filtered_kwargs)

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
        attn_imp = getattr(self, "attn_implementation", "flash_attention_2")
        if attn_imp == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("Warning: flash_attn package not installed. Falling back to sdpa.", flush=True)
                attn_imp = "sdpa"
        kwargs["attn_implementation"] = attn_imp
        return kwargs
