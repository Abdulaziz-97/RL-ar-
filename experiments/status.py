"""Experiment completion / resume detection."""

from __future__ import annotations

import json
from pathlib import Path


DONE_MARKER = "EXPERIMENT_DONE.json"


def _latest_trainer_state(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    # Prefer highest checkpoint number
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-", 1)[-1]) if p.name.split("-", 1)[-1].isdigit() else -1,
        reverse=True,
    )
    for d in checkpoints:
        state = d / "trainer_state.json"
        if state.exists():
            return state
    # Some trainers write state at output root
    root_state = output_dir / "trainer_state.json"
    return root_state if root_state.exists() else None


def read_global_step(output_dir: Path) -> int:
    state_path = _latest_trainer_state(output_dir)
    if state_path is None:
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", 0) or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def is_completed(output_dir: Path, expected_steps: int | None = None) -> bool:
    """
    True if experiment finished successfully.

    Prefer explicit DONE marker written by the runner after exit code 0.
    Fall back to trainer_state reaching expected_steps (or any positive step
    when expected_steps is None and a DONE marker is absent — conservative:
    only DONE or meeting expected_steps counts as complete for resume).
    """
    marker = output_dir / DONE_MARKER
    if marker.exists():
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("ok") is True:
                return True
        except (OSError, json.JSONDecodeError):
            pass

    step = read_global_step(output_dir)
    if expected_steps is not None:
        return step >= expected_steps
    return False


def write_done_marker(
    output_dir: Path,
    *,
    exp_id: str,
    returncode: int,
    cmd: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": returncode == 0,
        "exp_id": exp_id,
        "returncode": returncode,
        "cmd": cmd,
        "global_step": read_global_step(output_dir),
    }
    (output_dir / DONE_MARKER).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
