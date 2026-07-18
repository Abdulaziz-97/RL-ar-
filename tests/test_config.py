"""Tests for the SOTA config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from rlvr_sota.config import SOTAConfig


def test_default_config_values():
    config = SOTAConfig()
    assert config.loss_type == "dr_grpo"
    assert config.scale_rewards == "batch"
    assert config.beta == 0.04
    assert config.epsilon == 0.2
    assert config.epsilon_high == 0.28
    assert config.importance_sampling_level == "token"
    assert config.reward_weights == [0.6, 0.2, 0.05, 0.5, 0.3, 0.15]
    assert config.load_in_4bit is True
    assert config.num_generations == 8
    assert config.max_completion_length == 384
    assert config.stop_strings == ["</answer>"]
    assert config.temperature == 1.15
    assert config.enable_stability_callback is True
    assert config.entropy_collapse_action == "reduce_lr"
    assert config.enable_adaptive_beta is True
    assert config.stuck_beta == 0.02
    assert config.early_diag_steps == 30
    assert config.zero_variance_strategy == "direct_scoring"
    assert config.curriculum_schedule_type == "gaussian"
    assert config.enable_crps is True
    assert config.replay_buffer_size == 512
    assert config.sigma_fraction == 0.2


def test_from_yaml(tmp_path):
    yaml_content = {
        "model_name": "test/model",
        "loss_type": "dapo",
        "scale_rewards": "group",
        "beta": 0.0,
        "num_generations": 4,
        "load_in_4bit": False,
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(yaml_content, f)
    config = SOTAConfig.from_yaml(str(path))
    assert config.model_name == "test/model"
    assert config.loss_type == "dapo"
    assert config.scale_rewards == "group"
    assert config.beta == 0.0
    assert config.num_generations == 4
    assert config.load_in_4bit is False


def test_build_grpo_config():
    config = SOTAConfig(load_in_4bit=False, use_wandb=False, bf16=False)
    grpo_config = config.build_grpo_config()
    assert grpo_config.loss_type == "dr_grpo"
    assert grpo_config.scale_rewards == "batch"
    assert grpo_config.beta == 0.04
    assert grpo_config.epsilon == 0.2
    assert grpo_config.epsilon_high == 0.28
    assert grpo_config.importance_sampling_level == "token"
    assert grpo_config.reward_weights == [0.6, 0.2, 0.05, 0.5, 0.3, 0.15]
    assert grpo_config.num_generations == 8
    assert grpo_config.max_completion_length == 384
    assert "wandb" not in grpo_config.report_to


def test_build_grpo_config_scale_rewards_off():
    config = SOTAConfig(scale_rewards="off", load_in_4bit=False, use_wandb=False, bf16=False)
    grpo_config = config.build_grpo_config()
    assert grpo_config.scale_rewards in (False, "none", None)


def test_build_grpo_config_gspo_mode():
    config = SOTAConfig(
        importance_sampling_level="sequence",
        load_in_4bit=False,
        use_wandb=False,
        bf16=False,
    )
    grpo_config = config.build_grpo_config()
    assert grpo_config.importance_sampling_level == "sequence"


def test_build_grpo_config_gspo_token_mode():
    config = SOTAConfig(
        importance_sampling_level="sequence_token",
        load_in_4bit=False,
        use_wandb=False,
        bf16=False,
    )
    grpo_config = config.build_grpo_config()
    assert grpo_config.importance_sampling_level == "sequence_token"


def test_build_quantization_config():
    config = SOTAConfig(load_in_4bit=True)
    quant_config = config.build_quantization_config()
    assert quant_config is not None

    config_no_quant = SOTAConfig(load_in_4bit=False)
    assert config_no_quant.build_quantization_config() is None


def test_build_peft_config():
    config = SOTAConfig()
    peft_config = config.build_peft_config()
    assert peft_config.r == 64
    assert peft_config.lora_alpha == 128
    assert peft_config.target_modules is not None


def test_build_model_init_kwargs():
    config = SOTAConfig(load_in_4bit=True, bf16=True)
    kwargs = config.build_model_init_kwargs()
    assert "quantization_config" in kwargs
    assert "torch_dtype" in kwargs

    config_no_quant = SOTAConfig(load_in_4bit=False, bf16=True)
    kwargs = config_no_quant.build_model_init_kwargs()
    assert "quantization_config" not in kwargs


def test_load_example_yaml():
    import os
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "configs" / "qwen_4b_qlora.yaml"
    if not config_path.exists():
        pytest.skip("Example config not found")
    config = SOTAConfig.from_yaml(str(config_path))
    assert config.model_name == "Qwen/Qwen3.5-2B"
    assert config.loss_type == "dr_grpo"
    assert config.load_in_4bit is True
