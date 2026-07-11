"""
Trainer assembly — builds a GRPOTrainer with SOTA config + QLoRA + Arabic rewards
+ failure mining (RL-ZVP/POPO) + curriculum learning (E2H Gaussian) + CRPS.

Also provides cold-start SFT distillation before GRPO.
"""

from __future__ import annotations

from typing import Any, Optional

from rlvr_sota.config import SOTAConfig
from rlvr_sota.data import load_rlvr_dataset, load_cold_start_sft_dataset
from rlvr_sota.rewards import ALL_REWARD_FUNCS, DEFAULT_REWARD_WEIGHTS


def build_sft_trainer(
    config: SOTAConfig,
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

    peft_config = config.build_peft_config() if config.load_in_4bit and model is None else None
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

    return trainer


def build_trainer(
    config: SOTAConfig,
    train_dataset=None,
    eval_dataset=None,
    reward_funcs=None,
    reward_weights=None,
    model=None,
    processing_class=None,
):
    """Build a ready-to-train GRPOTrainer with failure mining + curriculum.

    This is Stage 2 of the pipeline. Run build_sft_trainer() first for cold-start.

    Args:
        config: SOTAConfig with all hyperparameters.
        train_dataset: Optional pre-loaded HF Dataset.
        eval_dataset: Optional pre-loaded eval Dataset.
        reward_funcs: Optional list of reward functions.
        reward_weights: Optional weights.
        model: Optional pre-loaded model.
        processing_class: Tokenizer/processor. Required when model is pre-loaded.
    """
    if reward_funcs is None:
        reward_funcs = ALL_REWARD_FUNCS
    if reward_weights is None:
        reward_weights = DEFAULT_REWARD_WEIGHTS

    config.reward_weights = reward_weights

    if train_dataset is None:
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
    peft_config = config.build_peft_config() if config.load_in_4bit and model is None else None

    grpo_config = config.build_grpo_config(include_model_init=(model is None))

    use_failure_mining = config.zero_variance_strategy != "discard" or config.enable_crps

    if use_failure_mining:
        from rlvr_sota.failure_mining_trainer import GRPOTrainerWithFailureMining as TrainerCls
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

    if use_failure_mining:
        trainer_kwargs["zero_variance_strategy"] = config.zero_variance_strategy
        trainer_kwargs["replay_buffer_size"] = config.replay_buffer_size
        trainer_kwargs["enable_crps"] = config.enable_crps
        trainer_kwargs["crps_max_age_steps"] = config.crps_max_age_steps
        trainer_kwargs["crps_max_traces"] = config.crps_max_traces

    print("Creating trainer (loading model)...", flush=True)
    trainer = TrainerCls(**trainer_kwargs)
    print(f"Trainer ready: {trainer.__class__.__name__}", flush=True)

    if config.sft_checkpoint_path and hasattr(trainer.model, "load_adapter"):
        from peft import PeftModel
        trainer.model = PeftModel.from_pretrained(
            trainer.model, config.sft_checkpoint_path, is_trainable=True
        )

    if (
        use_failure_mining
        and config.curriculum_schedule_type != "none"
        and hasattr(train_dataset, "column_names")
        and "difficulty_tag" in train_dataset.column_names
    ):
        trainer.attach_curriculum_sampler(config, train_dataset)

    return trainer

