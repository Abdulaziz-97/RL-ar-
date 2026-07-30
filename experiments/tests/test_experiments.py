"""Tests for experiments package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.builder import build_train_command, resolve_sft_checkpoint
from experiments.manifest import load_grid
from experiments.runner import select_experiments
from experiments.status import is_completed, write_done_marker


def test_load_grid_counts():
    m = load_grid()
    assert len(m.experiments) == 41
    assert len(m.phases()) == 10
    assert "C2" in m.sft


def test_baselines_explicitly_disable_sft():
    m = load_grid()
    b1 = m.by_id("B1")
    b2 = m.by_id("B2")
    assert b1.sft_override == ""
    assert b2.sft_override == ""
    assert resolve_sft_checkpoint(b1, Path("./runs/sft")) is None


def test_coldstart_inherits_global_sft():
    m = load_grid()
    c1 = m.by_id("C1")
    assert c1.sft_override is None
    sft = resolve_sft_checkpoint(c1, Path("./runs/sft"))
    assert sft == Path("./runs/sft")


def test_build_command_no_sft_for_baseline():
    m = load_grid()
    cmd = build_train_command(
        python="python",
        module="rlvr_pipeline",
        config=Path("configs/qwen_4b_qlora.yaml"),
        output_dir=Path("runs/B1"),
        exp=m.by_id("B1"),
        max_steps=10,
        global_sft=Path("./runs/sft"),
    )
    assert "--sft-checkpoint" not in cmd
    assert "--loss-type" in cmd and "grpo" in cmd
    assert "--disable-crps" in cmd
    assert "rlvr_pipeline" in cmd


def test_build_command_with_sft_and_crps():
    m = load_grid()
    cmd = build_train_command(
        python="python",
        module="rlvr_pipeline",
        config=Path("configs/qwen_4b_qlora.yaml"),
        output_dir=Path("runs/A1"),
        exp=m.by_id("A1"),
        max_steps=50,
        global_sft=Path("./runs/sft"),
        wandb_group="g",
    )
    assert cmd[cmd.index("--sft-checkpoint") + 1] == str(Path("./runs/sft"))
    assert "--enable-crps" in cmd
    assert "--curriculum" in cmd and "gaussian" in cmd
    assert "--wandb" in cmd


def test_hp_sweep_partial_flags():
    m = load_grid()
    cmd = build_train_command(
        python="python",
        module="rlvr_pipeline",
        config=Path("configs/qwen_4b_qlora.yaml"),
        output_dir=Path("runs/H2_T12"),
        exp=m.by_id("H2_T12"),
        max_steps=None,
        global_sft=None,
    )
    assert "--temperature" in cmd and "1.2" in cmd
    assert "--loss-type" not in cmd
    assert "--max-steps" not in cmd


def test_select_only_and_phase():
    m = load_grid()
    assert [e.exp_id for e in select_experiments(m, only="G2")] == ["G2"]
    phase3 = select_experiments(m, phase="3")
    assert [e.exp_id for e in phase3] == ["G1", "G2", "G3"]
    first = select_experiments(m, start_phase=1, max_phases=2)
    assert {e.phase_num for e in first} == {1, 2}


def test_manifest_does_not_mutate_on_reload():
    m1 = load_grid()
    _ = build_train_command(
        python="python",
        module="rlvr_pipeline",
        config=Path("c.yaml"),
        output_dir=Path("o"),
        exp=m1.by_id("B1"),
        max_steps=1,
        global_sft=None,
    )
    m2 = load_grid()
    assert m1.by_id("B1").flags == m2.by_id("B1").flags
    assert "comment" not in m1.by_id("B1").flags


def test_done_marker(tmp_path: Path):
    out = tmp_path / "B1"
    assert not is_completed(out, expected_steps=10)
    write_done_marker(out, exp_id="B1", returncode=0, cmd=["x"])
    assert is_completed(out)
    meta = json.loads((out / "EXPERIMENT_DONE.json").read_text(encoding="utf-8"))
    assert meta["ok"] is True


def test_invalid_flag_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "phases:\n  '1_x':\n    Z1:\n      loss_type: not_a_loss\n      comment: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_type"):
        load_grid(bad)


def test_cli_validate_and_dry_run():
    from experiments.__main__ import main

    assert main(["validate"]) == 0
    assert main(["list", "--phase", "1"]) == 0
    assert (
        main(
            [
                "run",
                "--dry-run",
                "--only",
                "B1",
                "--output",
                str(Path("runs/experiments_test_dry")),
                "--max-steps",
                "1",
            ]
        )
        == 0
    )
