"""Tests for StabilityCallback adaptive beta + early diagnostics."""

from types import SimpleNamespace

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rlvr_pipeline.stability_callback import StabilityCallback


def _args(lr=1e-5):
    return TrainingArguments(
        output_dir="./tmp_stability_test",
        learning_rate=lr,
        report_to=[],
    )


def _state(step=10):
    return TrainerState(global_step=step)


def test_format_collapse_warns_without_stopping():
    cb = StabilityCallback(consecutive=2, entropy_action="warn")
    control = TrainerControl()
    args = _args()

    cb.on_log(
        args,
        _state(1),
        control,
        logs={
            "entropy": 0.5,
            "rewards/format_reward_func/mean": 0.0,
            "rewards/format_reward_func/std": 0.0,
        },
    )
    cb.on_log(
        args,
        _state(2),
        control,
        logs={
            "entropy": 0.5,
            "rewards/format_reward_func/mean": 0.0,
            "rewards/format_reward_func/std": 0.0,
        },
    )
    assert control.should_training_stop is False
    assert cb.dashboard.format_collapse_alerts >= 1


def test_entropy_zero_reduce_lr_then_stop():
    cb = StabilityCallback(consecutive=2, entropy_action="reduce_lr", lr_reduction_factor=0.5)
    control = TrainerControl()
    args = _args(lr=1e-5)
    opt = SimpleNamespace(param_groups=[{"lr": 1e-5}])
    cb.trainer = SimpleNamespace(optimizer=opt, lr_scheduler=None, beta=0.04, args=SimpleNamespace(beta=0.04))

    # Include `loss` so logs pass the optimizer-step filter (skips failure-mining stubs).
    for step in (1, 2):
        cb.on_log(args, _state(step), control, logs={"entropy": 0.0, "loss": 0.1})

    assert cb._lr_already_reduced is True
    assert args.learning_rate == 1e-5 * 0.5
    assert opt.param_groups[0]["lr"] == 1e-5 * 0.5
    assert control.should_training_stop is False

    for step in (3, 4):
        cb.on_log(args, _state(step), control, logs={"entropy": 0.0, "loss": 0.1})

    assert control.should_training_stop is True


def test_entropy_action_stop_immediate():
    cb = StabilityCallback(consecutive=2, entropy_action="stop")
    control = TrainerControl()
    args = _args()
    cb.on_log(args, _state(1), control, logs={"entropy": 0.0, "loss": 0.1})
    cb.on_log(args, _state(2), control, logs={"entropy": 0.0, "loss": 0.1})
    assert control.should_training_stop is True


def test_skips_incomplete_failure_mining_logs():
    """Entropy=0 stubs without loss/rewards must not trip reduce_lr."""
    cb = StabilityCallback(consecutive=2, entropy_action="reduce_lr")
    control = TrainerControl()
    args = _args(lr=1e-5)
    for step in (1, 2, 3):
        cb.on_log(args, _state(step), control, logs={"entropy": 0.0})
    assert cb._lr_already_reduced is False
    assert args.learning_rate == 1e-5
    assert control.should_training_stop is False


def test_adaptive_beta_reduces_when_kl_near_zero_and_entropy_healthy():
    cb = StabilityCallback(consecutive=2, stuck_beta=0.02, kl_near_zero_threshold=1e-3)
    control = TrainerControl()
    args = _args()
    trainer_ns = SimpleNamespace(beta=0.04, args=SimpleNamespace(beta=0.04))
    cb.trainer = trainer_ns

    cb.on_log(args, _state(5), control, logs={"entropy": 0.7, "kl": 0.0, "loss": 0.1})
    assert trainer_ns.beta == 0.04  # need consecutive=2
    cb.on_log(args, _state(6), control, logs={"entropy": 0.7, "kl": 0.0, "loss": 0.1})
    assert trainer_ns.beta == 0.02
    assert trainer_ns.args.beta == 0.02
    assert cb._beta_already_reduced is True


def test_adaptive_beta_skipped_during_entropy_collapse():
    cb = StabilityCallback(consecutive=2, stuck_beta=0.02)
    control = TrainerControl()
    args = _args()
    trainer_ns = SimpleNamespace(beta=0.04, args=SimpleNamespace(beta=0.04))
    cb.trainer = trainer_ns

    # Entropy exactly 0 → go through entropy-zero path, not beta unlock.
    cb.on_log(args, _state(1), control, logs={"entropy": 0.0, "kl": 0.0, "loss": 0.1})
    cb.on_log(args, _state(2), control, logs={"entropy": 0.0, "kl": 0.0, "loss": 0.1})
    assert trainer_ns.beta == 0.04
    assert cb._beta_already_reduced is False


def test_adaptive_temperature_bumps_when_entropy_below_target():
    cb = StabilityCallback(
        enable_adaptive_temperature=True,
        entropy_target_min=2.0,
        temp_bump=0.15,
        temp_max=1.6,
        temp_bump_cooldown_steps=3,
        enable_adaptive_beta=False,
    )
    control = TrainerControl()
    args = _args()
    gen_cfg = SimpleNamespace(temperature=1.35)
    trainer_ns = SimpleNamespace(
        temperature=1.35,
        args=SimpleNamespace(temperature=1.35),
        generation_config=gen_cfg,
        generation_kwargs={"temperature": 1.35},
    )
    cb.trainer = trainer_ns

    cb.on_log(args, _state(10), control, logs={"entropy": 1.5, "loss": 0.1})
    assert trainer_ns.temperature == 1.5
    assert trainer_ns.args.temperature == 1.5
    assert gen_cfg.temperature == 1.5
    assert trainer_ns.generation_kwargs["temperature"] == 1.5
    assert cb._temp_bumps == 1

    # Cooldown: no second bump at step 11
    cb.on_log(args, _state(11), control, logs={"entropy": 1.2, "loss": 0.1})
    assert trainer_ns.temperature == 1.5
    assert cb._temp_bumps == 1

    # After cooldown, bump again up to max
    cb.on_log(args, _state(14), control, logs={"entropy": 1.0, "loss": 0.1})
    assert abs(trainer_ns.temperature - 1.6) < 1e-9
    assert cb._temp_bumps == 2


def test_early_diag_forces_log_for_first_n_steps():
    cb = StabilityCallback(early_diag_steps=30)
    control = TrainerControl()
    args = _args()

    cb.on_step_end(args, _state(1), control)
    assert control.should_log is True

    control = TrainerControl()
    cb.on_step_end(args, _state(30), control)
    assert control.should_log is True

    control = TrainerControl()
    cb.on_step_end(args, _state(31), control)
    assert control.should_log is not True
