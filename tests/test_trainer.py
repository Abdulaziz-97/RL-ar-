"""Integration tests for the RLVR GRPOTrainer pipeline with a tiny model.

These tests verify the full training loop runs end-to-end:
  data â†’ generate G completions â†’ score rewards â†’ compute advantages â†’ loss â†’ backprop
"""

import math

import pytest

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.data import load_rlvr_dataset
from rlvr_pipeline.trainer import build_trainer


def _tiny_config(tmp_path, **overrides):
    defaults = dict(
        model_name="tiny-test",
        load_in_4bit=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        use_transformers_continuous_batching=False,
        num_generations=2,
        max_completion_length=32,
        temperature=0.9,
        top_p=1.0,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=2,
        logging_steps=1,
        save_steps=1000,
        eval_steps=None,
        loss_type="dr_grpo",
        scale_rewards="group",
        beta=0.0,
        epsilon=0.2,
        epsilon_high=None,
        importance_sampling_level="token",
        mask_truncated_completions=False,
        log_completions=False,
        use_wandb=False,
        report_to="none",
        output_dir=str(tmp_path / "outputs"),
        optim="adamw_torch",
        zero_variance_strategy="discard",
        enable_crps=False,
        curriculum_schedule_type="none",
        stop_strings=None,
        enable_stability_callback=False,
    )
    defaults.update(overrides)
    return RLVRConfig(**defaults)


def test_trainer_constructs_with_tiny_model(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path)
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    assert trainer is not None
    assert trainer.model is tiny_model


def test_trainer_runs_training_steps(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path)
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )

    result = trainer.train()

    assert result is not None
    assert result.global_step > 0
    assert not math.isnan(result.training_loss) if result.training_loss else True


def test_trainer_reward_functions_called(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path)
    train_ds = load_rlvr_dataset(math_jsonl)

    call_count = [0]
    original_correctness = None
    from rlvr_pipeline.rewards import correctness_reward_func

    def counting_correctness(prompts, completions, **kwargs):
        call_count[0] += 1
        return correctness_reward_func(prompts, completions, **kwargs)

    from rlvr_pipeline.rewards import (
        format_reward_func,
        language_reward_func,
        answer_leak_penalty_func,
        structural_leak_penalty_func,
        length_penalty_func,
    )

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
        reward_funcs=[
            counting_correctness,
            format_reward_func,
            language_reward_func,
            answer_leak_penalty_func,
            structural_leak_penalty_func,
            length_penalty_func,
        ],
    )
    trainer.train()
    assert call_count[0] > 0


def test_trainer_gspo_mode(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, importance_sampling_level="sequence")
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    result = trainer.train()
    assert result.global_step > 0


def test_trainer_dapo_loss(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, loss_type="dapo")
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    result = trainer.train()
    assert result.global_step > 0


def test_trainer_saves_checkpoint(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, save_steps=1, max_steps=1)
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    trainer.train()
    trainer.save_model(str(tmp_path / "checkpoint"))

    import os
    saved = os.listdir(str(tmp_path / "checkpoint"))
    assert len(saved) > 0


def test_trainer_failure_mining_direct_scoring(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, zero_variance_strategy="direct_scoring", enable_crps=False)
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    result = trainer.train()
    assert result.global_step > 0


def test_trainer_curriculum_preserves_grpo_groups(
    tiny_model, tiny_tokenizer, math_jsonl, tmp_path
):
    config = _tiny_config(
        tmp_path,
        zero_variance_strategy="direct_scoring",
        curriculum_schedule_type="gaussian",
        max_steps=1,
    )
    train_ds = load_rlvr_dataset(math_jsonl)
    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    result = trainer.train()
    assert result.global_step == 1


def test_trainer_rejects_unimplemented_replay_buffer(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, zero_variance_strategy="replay_buffer", enable_crps=False)
    train_ds = load_rlvr_dataset(math_jsonl)

    with pytest.raises(ValueError, match="not implemented safely"):
        build_trainer(
            config=config,
            train_dataset=train_ds,
            model=tiny_model,
            processing_class=tiny_tokenizer,
        )


def test_trainer_failure_mining_crps_enabled(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(tmp_path, zero_variance_strategy="discard", enable_crps=True)
    train_ds = load_rlvr_dataset(math_jsonl)

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    result = trainer.train()
    assert result.global_step > 0


def test_trainer_with_curriculum(tiny_model, tiny_tokenizer, math_jsonl, tmp_path):
    config = _tiny_config(
        tmp_path,
        zero_variance_strategy="discard",
        curriculum_schedule_type="gaussian",
        sigma_fraction=0.3,
    )
    train_ds = load_rlvr_dataset(math_jsonl)
    assert "difficulty_tag" in train_ds.column_names

    trainer = build_trainer(
        config=config,
        train_dataset=train_ds,
        model=tiny_model,
        processing_class=tiny_tokenizer,
    )
    assert trainer._curriculum_sampler is not None
    result = trainer.train()
    assert result.global_step > 0
