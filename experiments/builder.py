"""Build train / SFT CLI commands from experiment specs (no mutation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.manifest import ExperimentSpec, SftSpec

FLAG_TO_CLI: dict[str, str] = {
    "loss_type": "--loss-type",
    "importance_sampling_level": "--importance-sampling",
    "scale_rewards": "--scale-rewards",
    "curriculum_schedule_type": "--curriculum",
    "zero_variance_strategy": "--zero-variance",
    "num_generations": "--num-generations",
    "temperature": "--temperature",
    "sigma_fraction": "--sigma-fraction",
    "beta": "--beta",
    "epsilon": "--epsilon",
    "epsilon_high": "--epsilon-high",
    "learning_rate": "--learning-rate",
}


def resolve_sft_checkpoint(
    exp: ExperimentSpec,
    global_sft: Path | str | None,
) -> Path | None:
    """
    Resolve SFT checkpoint for an experiment.

    - Explicit empty override (baselines B1/B2): no SFT
    - Explicit path override: use that path
    - No override: use global default (may be None)
    """
    if exp.sft_override is not None:
        if exp.sft_override == "":
            return None
        return Path(exp.sft_override)
    if global_sft is None or str(global_sft).strip() == "":
        return None
    return Path(global_sft)


def build_train_command(
    *,
    python: str,
    module: str,
    config: Path,
    output_dir: Path,
    exp: ExperimentSpec,
    max_steps: int | None,
    global_sft: Path | str | None,
    wandb_group: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build `python -m <module> train ...` argv. Never mutates `exp`."""
    cmd: list[str] = [
        python,
        "-m",
        module,
        "train",
        "--config",
        str(config),
        "--output",
        str(output_dir),
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]

    sft = resolve_sft_checkpoint(exp, global_sft)
    if sft is not None:
        cmd += ["--sft-checkpoint", str(sft)]

    if wandb_group:
        cmd += ["--wandb", "--wandb-group", wandb_group]

    for key, cli_flag in FLAG_TO_CLI.items():
        if key in exp.flags and exp.flags[key] is not None:
            cmd += [cli_flag, str(exp.flags[key])]

    if "enable_crps" in exp.flags and exp.flags["enable_crps"] is not None:
        cmd.append("--enable-crps" if exp.flags["enable_crps"] else "--disable-crps")

    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_sft_command(
    *,
    python: str,
    module: str,
    config: Path,
    spec: SftSpec,
    project_root: Path,
) -> list[str]:
    """Build SFT stage command from grid `sft` section."""
    output = Path(spec.output)
    if not output.is_absolute():
        output = project_root / output
    cmd: list[str] = [
        python,
        "-m",
        module,
        "sft",
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if spec.coldstart:
        cold = Path(spec.coldstart)
        if not cold.is_absolute():
            cold = project_root / cold
        cmd += ["--data", str(cold)]
    if spec.learning_rate is not None:
        cmd += ["--learning-rate", str(spec.learning_rate)]
    return cmd


def command_digest(cmd: list[str]) -> dict[str, Any]:
    """Stable summary for logging / dry-run."""
    return {"argv": cmd, "joined": " ".join(cmd)}
