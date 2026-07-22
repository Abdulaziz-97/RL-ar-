"""Trainer assembly: SFT cold-start + GRPO with QLoRA, Arabic rewards, curriculum, CRPS."""

from __future__ import annotations

from typing import Any, Optional

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.data import load_rlvr_dataset, load_cold_start_sft_dataset
from rlvr_pipeline.rewards import ALL_REWARD_FUNCS, DEFAULT_REWARD_WEIGHTS

# Qwen3.5 injects a pre-closed empty <think></think> into generation prompts.
# That makes reward_format unearnable; strip it so rollouts match SFT targets.
_EMPTY_THINK_INJECTION = "{{- '<think>\\n\\n</think>\\n\\n' }}"


def strip_think_injection(template: str | None) -> str | None:
    """Remove the pre-closed empty <think> block from a chat template string."""
    if not template or _EMPTY_THINK_INJECTION not in template:
        return template
    return template.replace(_EMPTY_THINK_INJECTION, "")


def load_sft_adapter_strict(peft_model, ckpt_dir: str) -> None:
    """Load SFT LoRA into an existing PEFT model, remapping Qwen3.5 key prefixes.

    Checkpoints may use ``model.language_model.layers.*`` while GRPO loads
    ``model.layers.*``. PEFT warns and silently loads nothing on mismatch;
    this remaps keys and fails hard if anything is left unmatched.
    """
    from pathlib import Path

    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    ckpt = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    live_keys = set(get_peft_model_state_dict(peft_model).keys())

    remapped = {}
    for key, value in ckpt.items():
        candidates = (key, key.replace(".language_model.", "."))
        target = next((c for c in candidates if c in live_keys), None)
        if target is None:
            raise RuntimeError(f"SFT adapter key cannot be mapped onto live model: {key}")
        remapped[target] = value

    missing = live_keys - set(remapped)
    if missing:
        raise RuntimeError(
            f"SFT adapter checkpoint is missing {len(missing)} keys, e.g. {sorted(missing)[:3]}"
        )

    set_peft_model_state_dict(peft_model, remapped)
    print(f"SFT adapter loaded strictly: {len(remapped)} tensors from {ckpt_dir}", flush=True)


def fix_chat_template(processing_class) -> bool:
    """Patch a tokenizer/processor in place. Returns True if a fix was applied."""
    if processing_class is None:
        return False
    tok = processing_class
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    template = getattr(tok, "chat_template", None)
    fixed = strip_think_injection(template)
    if fixed == template:
        return False
    tok.chat_template = fixed
    if tok is not processing_class and hasattr(processing_class, "chat_template"):
        processing_class.chat_template = fixed
    return True


def _attach_stop_string_criteria(trainer, stop_strings: list[str]) -> None:
    """Stop at </answer> via StopStringCriteria (TRL generate has no tokenizer=)."""
    if not stop_strings:
        return

    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        StopStringCriteria,
    )

    tok = getattr(trainer, "processing_class", None) or getattr(trainer, "_tokenizer", None)
    if tok is None:
        return
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    criteria = StoppingCriteriaList(
        [StopStringCriteria(tokenizer=tok, stop_strings=stop_strings)]
    )

    gen_config = getattr(trainer, "generation_config", None)
    if gen_config is not None and getattr(gen_config, "stop_strings", None):
        gen_config.stop_strings = None

    targets = [trainer.model]
    base_getter = getattr(trainer.model, "get_base_model", None)
    if callable(base_getter):
        try:
            base = base_getter()
            if base is not None and base not in targets:
                targets.append(base)
        except Exception:
            pass

    for model in targets:
        if getattr(model, "_rlvr_answer_stop_patched", False):
            continue
        original_generate = model.generate

        def generate_with_answer_stop(*args, _original=original_generate, **kwargs):
            existing = kwargs.get("stopping_criteria")
            if existing is None:
                kwargs["stopping_criteria"] = criteria
            else:
                kwargs["stopping_criteria"] = StoppingCriteriaList(
                    list(existing) + list(criteria)
                )
            gc = kwargs.get("generation_config")
            if gc is not None and getattr(gc, "stop_strings", None):
                gc.stop_strings = None
            return _original(*args, **kwargs)

        model.generate = generate_with_answer_stop
        model._rlvr_answer_stop_patched = True


def _align_trainable_dtype_for_amp(trainer, *, fp16: bool, bf16: bool) -> None:
    """Make QLoRA trainable weights GradScaler-safe on RTX 20-series.

    Qwen3.5 adapters often inherit bf16. With ``fp16=True`` GradScaler then either:
      - fails on bf16 grads (no CUDA unscale kernel), or
      - fails after casting adapters to fp16 ("Attempting to unscale FP16 gradients").
    Keep LoRA in float32 and disable AMP for this path.
    """
    model = getattr(trainer, "model", None)
    if model is None or bf16:
        return
    if not fp16:
        return

    import torch

    cast = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)
            cast += 1

    args = getattr(trainer, "args", None)
    if args is not None:
        args.fp16 = False
        args.bf16 = False

    print(
        f"QLoRA AMP fix: cast {cast} trainable params to float32; disabled fp16/bf16 GradScaler",
        flush=True,
    )


def build_sft_trainer(
    config: RLVRConfig,
    sft_dataset=None,
    model=None,
    processing_class=None,
):
    """Build a TRL SFTTrainer for cold-start CoT distillation.

    Sets a minimal chat template so Qwen3.5's thinking-mode tokens don't
    override the <think>/<answer> XML format we're teaching.
    """
    from trl import SFTConfig, SFTTrainer

    if sft_dataset is None:
        if not config.coldstart_data_path:
            raise ValueError("config.coldstart_data_path must be set for SFT stage")
        sft_dataset = load_cold_start_sft_dataset(
            config.coldstart_data_path,
            system_prompt=config.system_prompt,
        )

    peft_config = config.build_peft_config() if model is None else None
    model_init_kwargs = config.build_model_init_kwargs() if model is None else None

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim=config.optim,
        seed=config.seed,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=config.report_to if config.use_wandb else "none",
        dataset_text_field=None,
    )
    if config.max_steps is not None:
        sft_config.max_steps = config.max_steps
    if model_init_kwargs:
        sft_config.model_init_kwargs = model_init_kwargs

    trainer = SFTTrainer(
        model=model if model is not None else config.model_name,
        args=sft_config,
        train_dataset=sft_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )

    if fix_chat_template(getattr(trainer, "processing_class", None)):
        print("Chat template fixed: removed empty <think> injection (SFT)", flush=True)

    _align_trainable_dtype_for_amp(trainer, fp16=config.fp16, bf16=config.bf16)
    return trainer


def build_trainer(
    config: RLVRConfig,
    train_dataset=None,
    eval_dataset=None,
    reward_funcs=None,
    reward_weights=None,
    model=None,
    processing_class=None,
    evaluation_only: bool = False,
):
    """Build GRPOTrainer (Stage 2). Run build_sft_trainer() first for cold-start."""
    if reward_funcs is None:
        reward_funcs = ALL_REWARD_FUNCS
    if reward_weights is None:
        reward_weights = config.reward_weights if config.reward_weights else DEFAULT_REWARD_WEIGHTS
    config.reward_weights = reward_weights

    if train_dataset is None and not evaluation_only:
        if not config.train_data_path:
            raise ValueError("Either train_dataset or config.train_data_path must be set")
        print("Loading training data...", flush=True)
        train_dataset = load_rlvr_dataset(
            config.train_data_path,
            system_prompt=config.system_prompt,
        )
        print(f"Loaded {len(train_dataset)} training samples", flush=True)

    if eval_dataset is None and config.eval_data_path:
        eval_dataset = load_rlvr_dataset(
            config.eval_data_path,
            system_prompt=config.system_prompt,
        )

    print("Building GRPO config...", flush=True)

    model_init_kwargs = config.build_model_init_kwargs() if model is None else None
    peft_config = config.build_peft_config() if model is None else None

    grpo_config = config.build_grpo_config(include_model_init=(model is None))

    use_extended_trainer = (
        config.zero_variance_strategy != "discard"
        or config.enable_crps
        or config.curriculum_schedule_type != "none"
    )

    if config.importance_sampling_level == "sequence_token" and use_extended_trainer:
        raise ValueError(
            "sequence_token is incompatible with direct scoring, CRPS, and the "
            "custom curriculum; set zero_variance_strategy='discard', "
            "enable_crps=false, and curriculum_schedule_type='none'"
        )

    if use_extended_trainer:
        from rlvr_pipeline.failure_mining_trainer import GRPOTrainerWithFailureMining as TrainerCls
    elif config.importance_sampling_level == "sequence_token":
        try:
            from trl.experimental.gspo_token import GRPOTrainer as TrainerCls
        except ImportError as e:
            raise ImportError(
                "GSPO-token requires trl.experimental.gspo_token. Ensure TRL is up to date."
            ) from e
    else:
        from trl import GRPOTrainer as TrainerCls

    trainer_kwargs: dict[str, Any] = dict(
        model=model if model is not None else config.model_name,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        peft_config=peft_config,
        processing_class=processing_class,
    )

    if use_extended_trainer:
        trainer_kwargs["zero_variance_strategy"] = config.zero_variance_strategy
        trainer_kwargs["enable_crps"] = config.enable_crps
        trainer_kwargs["crps_max_age_steps"] = config.crps_max_age_steps
        trainer_kwargs["crps_max_traces"] = config.crps_max_traces

    print("Creating trainer (loading model)...", flush=True)
    trainer = TrainerCls(**trainer_kwargs)
    print(f"Trainer ready: {trainer.__class__.__name__}", flush=True)

    if fix_chat_template(getattr(trainer, "processing_class", None)):
        print("Chat template fixed: removed empty <think> injection (GRPO rollouts)", flush=True)

    # Do not PeftModel.from_pretrained on an existing PeftModel (silent no-load).
    if config.sft_checkpoint_path:
        load_sft_adapter_strict(trainer.model, config.sft_checkpoint_path)

    _align_trainable_dtype_for_amp(trainer, fp16=config.fp16, bf16=config.bf16)

    if config.stop_strings:
        _attach_stop_string_criteria(trainer, config.stop_strings)

    if (
        use_extended_trainer
        and config.curriculum_schedule_type != "none"
        and hasattr(train_dataset, "column_names")
        and "difficulty_tag" in train_dataset.column_names
    ):
        trainer.attach_curriculum_sampler(config, train_dataset)

    if config.enable_stability_callback:
        from rlvr_pipeline.stability_callback import StabilityCallback

        stability_cb = StabilityCallback(
            consecutive=config.stability_consecutive,
            entropy_action=config.entropy_collapse_action,
            lr_reduction_factor=config.entropy_lr_reduction_factor,
            enable_adaptive_beta=config.enable_adaptive_beta,
            kl_near_zero_threshold=config.kl_near_zero_threshold,
            stuck_beta=config.stuck_beta,
            enable_adaptive_temperature=config.enable_adaptive_temperature,
            entropy_target_min=config.entropy_target_min,
            temp_bump=config.temp_bump,
            temp_max=config.temp_max,
            temp_bump_cooldown_steps=config.temp_bump_cooldown_steps,
            early_diag_steps=config.early_diag_steps,
        )
        stability_cb.trainer = trainer
        trainer.add_callback(stability_cb)

    return trainer

