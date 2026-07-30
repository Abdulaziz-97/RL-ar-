"""Execute experiments from the grid with logging, retries, and resume."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.builder import build_sft_command, build_train_command, command_digest
from experiments.manifest import ExperimentSpec, GridManifest, load_grid
from experiments.status import is_completed, write_done_marker

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tee_run(cmd: list[str], log_file: Path, cwd: Path) -> int:
    """Run subprocess, stream stdout/stderr to console and append to log_file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 60}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"{'=' * 60}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def run_one(
    *,
    exp: ExperimentSpec,
    python: str,
    module: str,
    config: Path,
    base_output: Path,
    max_steps: int | None,
    global_sft: Path | None,
    wandb_group: str | None,
    project_root: Path,
    dry_run: bool = False,
    retries: int = 1,
    resume: bool = False,
) -> bool:
    output_dir = base_output / exp.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume and is_completed(output_dir, expected_steps=max_steps):
        print(f"  [{exp.exp_id}] SKIP (completed)")
        return True

    cmd = build_train_command(
        python=python,
        module=module,
        config=config,
        output_dir=output_dir,
        exp=exp,
        max_steps=max_steps,
        global_sft=global_sft,
        wandb_group=wandb_group,
    )

    print(f"  [{exp.exp_id}] {exp.comment}")
    print(f"           {' '.join(cmd)}")

    if dry_run:
        digest_path = output_dir / "dry_run.json"
        digest_path.write_text(
            json.dumps(command_digest(cmd), indent=2) + "\n",
            encoding="utf-8",
        )
        return True

    log_file = output_dir / "run.log"
    ok = False
    for attempt in range(retries + 1):
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n--- attempt {attempt + 1}/{retries + 1} ---\n")

        start = time.time()
        rc = _tee_run(cmd, log_file, project_root)
        elapsed = time.time() - start
        if rc == 0:
            write_done_marker(output_dir, exp_id=exp.exp_id, returncode=rc, cmd=cmd)
            print(f"  [{exp.exp_id}] OK ({elapsed / 60:.1f}m)")
            ok = True
            break
        print(f"  [{exp.exp_id}] FAIL attempt {attempt + 1} rc={rc} ({elapsed / 60:.1f}m)")
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\nFAILED attempt {attempt + 1} returncode={rc}\n")

    if not ok:
        write_done_marker(output_dir, exp_id=exp.exp_id, returncode=1, cmd=cmd)
    return ok


def run_sft(
    *,
    manifest: GridManifest,
    python: str,
    module: str,
    config: Path,
    project_root: Path,
    dry_run: bool = False,
    sft_id: str = "C2",
) -> bool:
    if sft_id not in manifest.sft:
        raise KeyError(f"No SFT spec '{sft_id}' in grid")
    spec = manifest.sft[sft_id]
    cmd = build_sft_command(
        python=python,
        module=module,
        config=config,
        spec=spec,
        project_root=project_root,
    )
    print(f"[SFT {sft_id}] {' '.join(cmd)}")
    if dry_run:
        return True
    log_dir = Path(spec.output)
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    rc = _tee_run(cmd, log_dir / "sft_run.log", project_root)
    return rc == 0


def select_experiments(
    manifest: GridManifest,
    *,
    only: str | None = None,
    phase: str | None = None,
    start_phase: int = 1,
    max_phases: int = 99,
) -> list[ExperimentSpec]:
    """Filter experiments by id / phase / phase range. Order follows the grid."""
    if only:
        exp = manifest.by_id(only)
        if phase:
            phase_ok = exp.phase == phase or (
                phase.isdigit() and exp.phase_num == int(phase)
            )
            if not phase_ok:
                raise ValueError(f"Experiment {only} is in phase {exp.phase}, not {phase}")
        return [exp]

    pool = list(manifest.experiments)

    if phase:
        if phase.isdigit():
            n = int(phase)
            pool = [e for e in pool if e.phase_num == n]
        else:
            pool = [e for e in pool if e.phase == phase]
        if not pool:
            raise ValueError(f"No experiments for phase={phase!r}")
        return pool

    pool = [e for e in pool if e.phase_num >= start_phase]
    # Keep first `max_phases` distinct phases in order
    kept_phases: list[str] = []
    selected: list[ExperimentSpec] = []
    for exp in pool:
        if exp.phase not in kept_phases:
            if len(kept_phases) >= max_phases:
                break
            kept_phases.append(exp.phase)
        if exp.phase in kept_phases:
            selected.append(exp)
    return selected


def write_summary(base_output: Path, results: dict[str, dict[str, bool]]) -> Path:
    path = base_output / "SUMMARY.json"
    base_output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    txt = base_output / "SUMMARY.txt"
    lines = ["Experiment Results", "=" * 60]
    for pn, r in results.items():
        lines.append(f"\n{pn}:")
        for eid, ok in r.items():
            lines.append(f"  {eid}: {'PASS' if ok else 'FAIL'}")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_grid(
    *,
    grid_path: Path | None = None,
    config: Path,
    output: Path,
    sft_checkpoint: Path | None,
    python: str | None = None,
    module: str = "rlvr_pipeline",
    max_steps: int | None = 300,
    wandb_group: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    retries: int = 1,
    only: str | None = None,
    phase: str | None = None,
    start_phase: int = 1,
    max_phases: int = 99,
    project_root: Path | None = None,
    run_sft_first: bool = False,
) -> dict[str, dict[str, bool]]:
    root = project_root or PROJECT_ROOT
    py = python or sys.executable
    manifest = load_grid(grid_path)

    if run_sft_first:
        ok = run_sft(
            manifest=manifest,
            python=py,
            module=module,
            config=config if config.is_absolute() else root / config,
            project_root=root,
            dry_run=dry_run,
        )
        if not ok and not dry_run:
            raise RuntimeError("SFT stage failed; aborting RL experiments")

    selected = select_experiments(
        manifest,
        only=only,
        phase=phase,
        start_phase=start_phase,
        max_phases=max_phases,
    )
    if not selected:
        raise ValueError("No experiments selected (check --only / --phase / --start-phase)")

    cfg = config if config.is_absolute() else root / config
    out = output if output.is_absolute() else root / output
    sft = None
    if sft_checkpoint is not None and str(sft_checkpoint).strip():
        sft = sft_checkpoint if Path(sft_checkpoint).is_absolute() else root / sft_checkpoint

    results: dict[str, dict[str, bool]] = {}
    current_phase = None
    for exp in selected:
        if exp.phase != current_phase:
            current_phase = exp.phase
            print(f"\n{'#' * 60}\n# Phase: {current_phase}\n{'#' * 60}")
            results.setdefault(current_phase, {})

        ok = run_one(
            exp=exp,
            python=py,
            module=module,
            config=cfg,
            base_output=out,
            max_steps=max_steps,
            global_sft=sft,
            wandb_group=wandb_group,
            project_root=root,
            dry_run=dry_run,
            retries=retries,
            resume=resume,
        )
        results[current_phase][exp.exp_id] = ok

    write_summary(out, results)
    return results
