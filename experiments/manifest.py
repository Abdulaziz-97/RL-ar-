"""Load and validate the experiment grid without mutating it."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Keys that are metadata / handled specially — not passed as algorithm CLI flags.
META_KEYS = frozenset({"comment", "sft_checkpoint_path"})

ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "loss_type": frozenset({"grpo", "dapo", "dr_grpo", "sapo", "bnpo"}),
    "importance_sampling_level": frozenset({"token", "sequence", "sequence_token"}),
    "scale_rewards": frozenset({"group", "batch", "off"}),
    "curriculum_schedule_type": frozenset({"gaussian", "fixed_switch", "random_mix", "none"}),
    "zero_variance_strategy": frozenset({"direct_scoring", "replay_buffer", "discard"}),
}

KNOWN_FLAGS = frozenset(
    {
        "loss_type",
        "importance_sampling_level",
        "scale_rewards",
        "curriculum_schedule_type",
        "zero_variance_strategy",
        "enable_crps",
        "num_generations",
        "temperature",
        "sigma_fraction",
        "beta",
        "epsilon",
        "epsilon_high",
        "learning_rate",
        "sft_checkpoint_path",
        "comment",
    }
)

DEFAULT_GRID = Path(__file__).resolve().parent / "grid.yaml"


@dataclass(frozen=True)
class ExperimentSpec:
    """One runnable experiment from the grid."""

    exp_id: str
    phase: str
    phase_num: int
    flags: dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    # None = use global default; "" = explicitly no SFT; path = use this.
    sft_override: str | None = None


@dataclass(frozen=True)
class SftSpec:
    exp_id: str
    coldstart: str
    output: str
    learning_rate: float | None = None


@dataclass(frozen=True)
class GridManifest:
    sft: dict[str, SftSpec]
    experiments: tuple[ExperimentSpec, ...]

    def by_id(self, exp_id: str) -> ExperimentSpec:
        for exp in self.experiments:
            if exp.exp_id == exp_id:
                return exp
        raise KeyError(f"Unknown experiment id: {exp_id}")

    def phases(self) -> list[str]:
        seen: list[str] = []
        for exp in self.experiments:
            if exp.phase not in seen:
                seen.append(exp.phase)
        return seen

    def for_phase(self, phase: str) -> list[ExperimentSpec]:
        return [e for e in self.experiments if e.phase == phase]


def _phase_num(phase_name: str) -> int:
    return int(str(phase_name).split("_", 1)[0])


def _normalize_flags(flags: dict[str, Any]) -> dict[str, Any]:
    """YAML maps bare `off`/`on`/`yes`/`no` to bools — restore intended strings."""
    out = dict(flags)
    # scale_rewards: off  →  False in PyYAML
    if "scale_rewards" in out and out["scale_rewards"] is False:
        out["scale_rewards"] = "off"
    if "scale_rewards" in out and out["scale_rewards"] is True:
        # unlikely; treat as invalid later unless user meant something else
        pass
    return out


def _validate_flags(exp_id: str, flags: dict[str, Any]) -> None:
    unknown = set(flags) - KNOWN_FLAGS
    if unknown:
        raise ValueError(f"{exp_id}: unknown flag(s): {sorted(unknown)}")

    for key, allowed in ALLOWED_VALUES.items():
        if key in flags and flags[key] is not None and flags[key] not in allowed:
            raise ValueError(
                f"{exp_id}: invalid {key}={flags[key]!r}; allowed={sorted(allowed)}"
            )

    if "enable_crps" in flags and flags["enable_crps"] is not None:
        if not isinstance(flags["enable_crps"], bool):
            raise ValueError(f"{exp_id}: enable_crps must be bool, got {type(flags['enable_crps'])}")

    for num_key in ("num_generations",):
        if num_key in flags and flags[num_key] is not None:
            if not isinstance(flags[num_key], int) or flags[num_key] < 1:
                raise ValueError(f"{exp_id}: {num_key} must be positive int")

    for float_key in ("temperature", "sigma_fraction", "beta", "epsilon", "epsilon_high", "learning_rate"):
        if float_key in flags and flags[float_key] is not None:
            try:
                float(flags[float_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{exp_id}: {float_key} must be numeric") from exc


def load_grid(path: Path | str | None = None) -> GridManifest:
    """Load grid.yaml into an immutable manifest. Validates every experiment."""
    grid_path = Path(path) if path else DEFAULT_GRID
    raw = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "phases" not in raw:
        raise ValueError(f"Invalid grid at {grid_path}: missing 'phases'")

    sft_specs: dict[str, SftSpec] = {}
    for sid, sflags in (raw.get("sft") or {}).items():
        sft_specs[sid] = SftSpec(
            exp_id=sid,
            coldstart=str(sflags.get("coldstart", "")),
            output=str(sflags.get("output", "./runs/sft")),
            learning_rate=sflags.get("learning_rate"),
        )

    experiments: list[ExperimentSpec] = []
    for phase_name in sorted(raw["phases"].keys(), key=_phase_num):
        phase_exps = raw["phases"][phase_name] or {}
        for exp_id, flags in phase_exps.items():
            flags = _normalize_flags(deepcopy(flags) if flags else {})
            _validate_flags(exp_id, flags)
            comment = str(flags.get("comment", "") or "")
            sft_override: str | None
            if "sft_checkpoint_path" in flags:
                sft_override = "" if flags["sft_checkpoint_path"] in (None, "") else str(flags["sft_checkpoint_path"])
            else:
                sft_override = None
            algo = {k: v for k, v in flags.items() if k not in META_KEYS}
            experiments.append(
                ExperimentSpec(
                    exp_id=exp_id,
                    phase=str(phase_name),
                    phase_num=_phase_num(phase_name),
                    flags=algo,
                    comment=comment,
                    sft_override=sft_override,
                )
            )

    return GridManifest(sft=sft_specs, experiments=tuple(experiments))
