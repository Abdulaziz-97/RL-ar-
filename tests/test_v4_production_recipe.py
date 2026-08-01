"""TDD gates for the V4 production hardening plan.

These tests define the production contract. Implement code until they pass.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from rlvr_pipeline.config import RLVRConfig

REPO = Path(__file__).resolve().parents[1]
V4_YAML = REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml"


# ---------------------------------------------------------------------------
# Config: stage separation & schema fidelity
# ---------------------------------------------------------------------------


def test_v4_yaml_exists():
    assert V4_YAML.exists()


def test_from_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model_name: x\nnot_a_real_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        RLVRConfig.from_yaml(path)


def test_from_yaml_keeps_eval_and_save_fields(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model_name: test/model
            eval_strategy: "no"
            save_strategy: steps
            save_total_limit: 5
            max_eval_samples: 32
            enable_benchmark_probe: false
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = RLVRConfig.from_yaml(path)
    assert cfg.eval_strategy == "no"
    assert cfg.save_strategy == "steps"
    assert cfg.save_total_limit == 5
    assert cfg.max_eval_samples == 32
    assert cfg.enable_benchmark_probe is False


def test_v4_sft_step_budget_independent_of_grpo_max_steps():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert cfg.max_steps == 250
    # SFT must not inherit the GRPO step cap.
    assert getattr(cfg, "sft_max_steps", None) in (None, 0) or cfg.sft_max_steps != 250
    assert getattr(cfg, "sft_num_train_epochs", None) is not None
    assert cfg.sft_num_train_epochs >= 1


def test_build_sft_config_ignores_grpo_max_steps():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    from rlvr_pipeline.trainer import build_sft_trainer

    # Inspect SFTConfig construction without loading a real model/dataset.
    with patch("rlvr_pipeline.trainer.load_cold_start_sft_dataset") as mock_ds, patch(
        "trl.SFTTrainer"
    ) as mock_trainer_cls, patch("trl.SFTConfig") as mock_sft_cfg, patch(
        "transformers.AutoModelForCausalLM"
    ), patch(
        "peft.get_peft_model", side_effect=lambda m, c: m
    ), patch(
        "peft.PeftModel"
    ):
        mock_ds.return_value = MagicMock()
        mock_sft_cfg.return_value = MagicMock(max_steps=None)
        mock_trainer_cls.return_value = MagicMock(model=MagicMock(), processing_class=None)
        cfg.instruction_base_model = None
        cfg.coldstart_data_path = str(REPO / "data" / "arabic_reasoning_coldstart_v5.jsonl")
        try:
            build_sft_trainer(cfg)
        except Exception:
            # Model download may fail offline; we only care about SFTConfig kwargs.
            pass
        assert mock_sft_cfg.called
        kwargs = mock_sft_cfg.call_args.kwargs
        # Must not pass GRPO's 250 into SFT max_steps.
        assert kwargs.get("max_steps") != 250


def test_v4_lora_targets_exclude_embed_and_lm_head():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    forbidden = {"embed_tokens", "lm_head"}
    assert forbidden.isdisjoint(set(cfg.lora_target_modules))


def test_v4_validate_rejects_embed_targets():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    cfg.lora_target_modules = list(cfg.lora_target_modules) + ["embed_tokens"]
    with pytest.raises(ValueError, match="embed_tokens|lm_head"):
        cfg.validate()


def test_v4_beta_anchored_and_adaptive_features_off():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert cfg.beta == pytest.approx(0.02)
    assert cfg.enable_adaptive_beta is False
    assert cfg.enable_adaptive_temperature is False
    assert cfg.enable_benchmark_probe is False
    assert cfg.curriculum_schedule_type == "gaussian"
    assert cfg.zero_variance_strategy == "direct_scoring"
    assert cfg.top_entropy_quantile == pytest.approx(0.2)
    assert cfg.push_checkpoints_to_hub is True
    assert cfg.hub_model_id == "aziz9788/qwen35-4b-arabic-rlvr-v5"
    assert cfg.hub_private is True
    assert cfg.delete_local_checkpoint_after_hub_push is True


def test_hub_checkpoint_callback_pushes_and_deletes_older(tmp_path):
    from rlvr_pipeline.hub_checkpoint_callback import HubCheckpointCallback

    out = tmp_path / "run"
    out.mkdir()
    for step in (50, 100, 150):
        d = out / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")

    cb = HubCheckpointCallback(
        hub_model_id="aziz9788/test-hub-ckpt",
        enabled=True,
        hub_private=True,
        delete_local_after_push=True,
        keep_local_last_n=2,
        token="hf_test_token",
    )

    with patch.object(cb, "_push_checkpoint", return_value=True) as mock_push:
        args = MagicMock(process_index=0, output_dir=str(out))
        # Simulate three successful saves; track pushed steps via real on_save path.
        for step in (50, 100, 150):
            state = MagicMock(global_step=step)
            cb.on_save(args, state, MagicMock())
        assert mock_push.call_count == 3

    # keep_local_last_n=2 → checkpoint-50 deleted; 100 and 150 remain.
    assert not (out / "checkpoint-50").exists()
    assert (out / "checkpoint-100").exists()
    assert (out / "checkpoint-150").exists()


def test_hub_checkpoint_callback_no_delete_on_push_failure(tmp_path):
    from rlvr_pipeline.hub_checkpoint_callback import HubCheckpointCallback

    out = tmp_path / "run"
    out.mkdir()
    d = out / "checkpoint-50"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")

    cb = HubCheckpointCallback(
        hub_model_id="aziz9788/test-hub-ckpt",
        delete_local_after_push=True,
        keep_local_last_n=1,
        token="hf_test_token",
    )
    with patch.object(cb, "_push_checkpoint", return_value=False):
        cb.on_save(
            MagicMock(process_index=0, output_dir=str(out)),
            MagicMock(global_step=50),
            MagicMock(),
        )
    assert d.exists()


def test_trainer_wires_hub_callback_when_enabled():
    src = (REPO / "src" / "rlvr_pipeline" / "trainer.py").read_text(encoding="utf-8")
    assert "_maybe_attach_hub_checkpoint_callback" in src
    assert "HubCheckpointCallback" in src



def test_v4_reward_weights_length_six_correctness_dominant():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert len(cfg.reward_weights) == 6
    assert cfg.reward_weights[0] >= cfg.reward_weights[1]
    # Length remains zero until calibrated.
    assert cfg.reward_weights[5] == 0.0


def test_build_grpo_config_no_mystery_210_default():
    cfg = RLVRConfig(max_steps=None, load_in_4bit=False, bf16=False, use_wandb=False)
    grpo = cfg.build_grpo_config(include_model_init=False)
    # Must not silently inject 210 when max_steps is unset.
    assert getattr(grpo, "max_steps", -1) in (-1, None, -1) or grpo.max_steps != 210


def test_build_grpo_config_passes_save_total_limit():
    cfg = RLVRConfig(
        load_in_4bit=False,
        bf16=False,
        use_wandb=False,
        save_total_limit=5,
        max_steps=10,
    )
    grpo = cfg.build_grpo_config(include_model_init=False)
    assert getattr(grpo, "save_total_limit", None) == 5


def test_entropy_regularization_fails_closed_when_trl_lacks_fields():
    cfg = RLVRConfig(entropy_coef=0.05, use_adaptive_entropy=True)
    with pytest.raises(ValueError, match="entropy regularization|does not accept"):
        cfg.validate()


def test_v4_yaml_entropy_disabled_for_installed_trl():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert cfg.entropy_coef == 0
    assert cfg.use_adaptive_entropy is False


def test_wandb_project_sets_env_when_use_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = RLVRConfig(use_wandb=True, wandb_project="arabic-reasoning-rlvr-v4")
    cfg.sync_wandb_env()
    assert os.environ["WANDB_PROJECT"] == "arabic-reasoning-rlvr-v4"
    grpo = cfg.build_grpo_config(include_model_init=False)
    assert "wandb" in grpo.report_to


# ---------------------------------------------------------------------------
# CLI: fresh by default, resume opt-in
# ---------------------------------------------------------------------------


def test_cli_train_defaults_to_fresh_no_auto_resume(tmp_path):
    from rlvr_pipeline import cli

    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoint-100").mkdir()
    (out / "checkpoint-100" / "adapter_config.json").write_text("{}", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump(
        {
            "model_name": "tiny",
            "train_data_path": str(tmp_path / "train.jsonl"),
            "output_dir": str(out),
            "max_steps": 2,
            "num_generations": 2,
            "per_device_train_batch_size": 2,
            "load_in_4bit": False,
            "bf16": False,
            "fp16": False,
            "use_wandb": False,
            "enable_crps": False,
            "curriculum_schedule_type": "none",
            "zero_variance_strategy": "discard",
            "enable_stability_callback": False,
            "enable_benchmark_probe": False,
        },
        cfg_path.open("w", encoding="utf-8"),
    )
    (tmp_path / "train.jsonl").write_text(
        '{"prompt":"q","answer_spec":{"type":"integer","canonical":1},'
        '"metadata":{"ground_truth_answer":"1"},"domain":"math"}\n',
        encoding="utf-8",
    )

    trainer = MagicMock()
    trainer.train_dataset = [0]
    trainer.get_train_dataloader = MagicMock(return_value=[0, 0])
    trainer.args = MagicMock(
        max_steps=2,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
    )

    with patch("rlvr_pipeline.cli.build_trainer", return_value=trainer):
        rc = cli.main(["train", "--config", str(cfg_path)])
    assert rc == 0
    trainer.train.assert_called_once()
    # Fresh default: no resume_from_checkpoint kwarg / positional.
    kwargs = trainer.train.call_args.kwargs
    assert kwargs.get("resume_from_checkpoint") in (None, False) or "resume_from_checkpoint" not in kwargs
    if trainer.train.call_args.args:
        assert trainer.train.call_args.args[0] is not True


def test_cli_train_resume_flag_uses_checkpoint(tmp_path):
    from rlvr_pipeline import cli

    out = tmp_path / "run"
    ckpt = out / "checkpoint-50"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "trainer_state.json").write_text('{"global_step": 50}', encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump(
        {
            "model_name": "tiny",
            "train_data_path": str(tmp_path / "train.jsonl"),
            "output_dir": str(out),
            "max_steps": 100,
            "num_generations": 2,
            "per_device_train_batch_size": 2,
            "load_in_4bit": False,
            "bf16": False,
            "fp16": False,
            "use_wandb": False,
            "enable_crps": False,
            "curriculum_schedule_type": "none",
            "zero_variance_strategy": "discard",
            "enable_stability_callback": False,
            "enable_benchmark_probe": False,
        },
        cfg_path.open("w", encoding="utf-8"),
    )
    (tmp_path / "train.jsonl").write_text(
        '{"prompt":"q","answer_spec":{"type":"integer","canonical":1},'
        '"metadata":{"ground_truth_answer":"1"},"domain":"math"}\n',
        encoding="utf-8",
    )

    trainer = MagicMock()
    trainer.train_dataset = [0]
    trainer.get_train_dataloader = MagicMock(return_value=[0])
    trainer.args = MagicMock(
        max_steps=100,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
    )

    with patch("rlvr_pipeline.cli.build_trainer", return_value=trainer):
        rc = cli.main(["train", "--config", str(cfg_path), "--resume"])
    assert rc == 0
    assert trainer.train.call_args.kwargs.get("resume_from_checkpoint") == str(ckpt)


# ---------------------------------------------------------------------------
# Failure mining: imports & zero-variance baseline
# ---------------------------------------------------------------------------


def test_failure_mining_module_imports_json_and_path():
    src = (REPO / "src" / "rlvr_pipeline" / "failure_mining_trainer.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            for alias in node.names:
                imported.add(alias.name)
    assert "json" in imported
    assert "Path" in imported or "pathlib" in imported


def test_v4_zero_variance_is_direct_scoring():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert cfg.zero_variance_strategy == "direct_scoring"


def test_build_grpo_config_passes_top_entropy_quantile():
    cfg = RLVRConfig.from_yaml(V4_YAML)
    grpo = cfg.build_grpo_config(include_model_init=False)
    assert grpo.top_entropy_quantile == pytest.approx(0.2)


def test_vllm_max_model_len_maps_to_trl_vllm_max_model_length():
    """TRL GRPOConfig uses vllm_max_model_length; YAML keeps vllm_max_model_len."""
    cfg = RLVRConfig.from_yaml(V4_YAML)
    assert cfg.use_vllm is False
    # Off: must not inject dead kwargs that get silently filtered.
    grpo_off = cfg.build_grpo_config(include_model_init=False)
    assert not hasattr(grpo_off, "vllm_max_model_len") or getattr(grpo_off, "vllm_max_model_len", None) in (None, 4096)

    cfg.use_vllm = True
    cfg.vllm_max_model_len = 8192
    cfg.vllm_gpu_memory_utilization = 0.35
    grpo = cfg.build_grpo_config(include_model_init=False)
    assert grpo.vllm_max_model_length == 8192
    assert grpo.vllm_gpu_memory_utilization == pytest.approx(0.35)
    assert grpo.use_vllm is True


# ---------------------------------------------------------------------------
# Probe / eval API contracts
# ---------------------------------------------------------------------------


def test_probe_callback_uses_checkpoint_flag_not_model_dir():
    from rlvr_pipeline.benchmark_probe_callback import BenchmarkProbeCallback

    with patch("subprocess.run") as mock_run:
        mock_run.reset_mock()
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        cb = BenchmarkProbeCallback(enable_probe=True, base_model="Qwen/Qwen3.5-4B")
        args = MagicMock(process_index=0, output_dir="./outputs/x")
        state = MagicMock(global_step=10)
        cb.on_save(args, state, MagicMock())
        cmd = mock_run.call_args[0][0]
        assert "--checkpoint" in cmd
        assert "--model-dir" not in cmd


def test_probe_disabled_skips_subprocess():
    from rlvr_pipeline.benchmark_probe_callback import BenchmarkProbeCallback

    with patch("subprocess.run") as mock_run:
        cb = BenchmarkProbeCallback(enable_probe=False)
        cb.on_save(MagicMock(process_index=0, output_dir="./o"), MagicMock(global_step=1), MagicMock())
        mock_run.assert_not_called()


def test_run_benchmark_probe_grading_helpers_compatible():
    """Probe must use the real generative_utils signatures."""
    from official_eval.tasks.araeval.generative_utils import (
        build_generative_prompt,
        grade_answer,
    )

    sig = inspect.signature(build_generative_prompt)
    params = list(sig.parameters)
    assert params[0] == "question"
    assert params[1] == "choices"
    # grade_answer(extracted, gold_index)
    gsig = inspect.signature(grade_answer)
    assert len(gsig.parameters) == 2


def test_probe_script_source_does_not_use_broken_apis():
    src = (REPO / "scripts" / "run_benchmark_probe.py").read_text(encoding="utf-8")
    # Broken patterns from the audit must be gone.
    assert 'docs[i]["target"]' not in src
    assert "grade_answer(extracted, docs[i][\"target\"], task_name)" not in src
    assert "build_generative_prompt(doc, task_name" not in src


# ---------------------------------------------------------------------------
# Orchestrator / shell contracts
# ---------------------------------------------------------------------------


def test_vastai_script_passes_grpo_output_explicitly():
    src = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")
    assert "--output" in src
    assert "GRPO_OUT" in src
    # Must not blanket-kill all python processes.
    assert "pkill -9 -f python" not in src


def test_master_orchestrator_does_not_cap_sft_at_100():
    path = REPO / "data_1" / "scripts" / "run_v4_master_orchestrator.py"
    if not path.exists():
        pytest.skip("orchestrator not present")
    src = path.read_text(encoding="utf-8")
    assert "--max-steps 100" not in src


def test_run_eval_sh_uses_instruction_base_not_raw_qwen():
    path = REPO / "run_eval.sh"
    if not path.exists():
        pytest.skip("run_eval.sh not present")
    src = path.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-4B" not in src or "T06" in src
    assert "T06" in src or "instruction" in src.lower()


# ---------------------------------------------------------------------------
# Trainer: strict SFT load is wired
# ---------------------------------------------------------------------------


def test_build_trainer_source_calls_load_sft_adapter_strict():
    src = (REPO / "src" / "rlvr_pipeline" / "trainer.py").read_text(encoding="utf-8")
    # GRPO path must use the strict loader, not only bare from_pretrained.
    assert "load_sft_adapter_strict(" in src
    # Rough guard: GRPO handoff section mentions strict load near sft_checkpoint.
    assert "sft_checkpoint_path" in src


def test_trainer_has_no_session_debug_agent_logs():
    src = (REPO / "src" / "rlvr_pipeline" / "trainer.py").read_text(encoding="utf-8")
    assert "a273d4" not in src
    assert "#region agent log" not in src


def test_probe_callback_has_no_session_debug_agent_logs():
    src = (REPO / "src" / "rlvr_pipeline" / "benchmark_probe_callback.py").read_text(
        encoding="utf-8"
    )
    assert "a273d4" not in src
    assert "#region agent log" not in src
