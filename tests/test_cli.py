"""Tests for the CLI argument parsing and entry point."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from rlvr_pipeline.cli import main, _build_arg_parser


def test_arg_parser_train_command():
    parser = _build_arg_parser()
    args = parser.parse_args(["train", "--config", "config.yaml", "--data", "data.jsonl"])
    assert args.command == "train"
    assert args.config == "config.yaml"
    assert args.data == "data.jsonl"


def test_arg_parser_train_with_overrides():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "train", "--config", "c.yaml", "--data", "d.jsonl",
        "--model", "Qwen/test", "--wandb", "--max-steps", "10",
    ])
    assert args.model == "Qwen/test"
    assert args.wandb is True
    assert args.max_steps == 10


def test_arg_parser_eval_command():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "eval", "--config", "c.yaml", "--data", "d.jsonl", "--checkpoint", "ckpt/"
    ])
    assert args.command == "eval"
    assert args.checkpoint == "ckpt/"


def test_main_missing_data_returns_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: test\n", encoding="utf-8")
    result = main(["train", "--config", str(config_path)])
    assert result == 1


def test_main_no_command_returns_error():
    with pytest.raises(SystemExit):
        main([])


def test_eval_uses_adapter_and_does_not_require_training_data(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: base-model\nload_in_4bit: false\n", encoding="utf-8")
    trainer = MagicMock()
    trainer.evaluate.return_value = {"reward": 1.0}

    with patch("rlvr_pipeline.cli.build_trainer", return_value=trainer) as build:
        result = main(
            [
                "eval",
                "--config",
                str(config_path),
                "--data",
                "eval.jsonl",
                "--checkpoint",
                "adapter/",
            ]
        )

    assert result == 0
    config = build.call_args.args[0]
    assert config.model_name == "base-model"
    assert config.sft_checkpoint_path == "adapter/"
    assert build.call_args.kwargs["evaluation_only"] is True
