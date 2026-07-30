"""CLI: python -m experiments <list|validate|run|sft>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.manifest import DEFAULT_GRID, load_grid
from experiments.runner import PROJECT_ROOT, run_grid, run_sft, select_experiments


def _cmd_list(args: argparse.Namespace) -> int:
    manifest = load_grid(args.grid)
    selected = select_experiments(
        manifest,
        only=args.only,
        phase=args.phase,
        start_phase=args.start_phase,
        max_phases=args.max_phases,
    )
    current = None
    for exp in selected:
        if exp.phase != current:
            current = exp.phase
            print(f"\n{current}")
        sft = "inherit" if exp.sft_override is None else ("none" if exp.sft_override == "" else exp.sft_override)
        print(f"  {exp.exp_id:10s}  sft={sft:8s}  {exp.comment}")
        if args.verbose:
            for k, v in sorted(exp.flags.items()):
                print(f"             {k}: {v}")
    print(f"\nTotal: {len(selected)} experiments")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_grid(args.grid)
    print(f"OK: {len(manifest.experiments)} experiments in {len(manifest.phases())} phases")
    for p in manifest.phases():
        n = len(manifest.for_phase(p))
        print(f"  {p}: {n}")
    return 0


def _cmd_sft(args: argparse.Namespace) -> int:
    manifest = load_grid(args.grid)
    ok = run_sft(
        manifest=manifest,
        python=args.python or sys.executable,
        module=args.module,
        config=Path(args.config),
        project_root=PROJECT_ROOT,
        dry_run=args.dry_run,
        sft_id=args.sft_id,
    )
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    results = run_grid(
        grid_path=Path(args.grid) if args.grid else None,
        config=Path(args.config),
        output=Path(args.output),
        sft_checkpoint=Path(args.sft_checkpoint) if args.sft_checkpoint else None,
        python=args.python,
        module=args.module,
        max_steps=args.max_steps,
        wandb_group=args.wandb_group if args.wandb else None,
        resume=args.resume,
        dry_run=args.dry_run,
        retries=args.retries,
        only=args.only,
        phase=args.phase,
        start_phase=args.start_phase,
        max_phases=args.max_phases,
        run_sft_first=args.run_sft_first,
    )
    flat = [ok for phase in results.values() for ok in phase.values()]
    failed = sum(1 for ok in flat if not ok)
    print(f"\nDone: {len(flat) - failed}/{len(flat)} passed")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Run Arabic RLVR experiment grid without mutating the YAML.",
    )
    p.add_argument(
        "--grid",
        default=str(DEFAULT_GRID),
        help="Path to grid.yaml (default: experiments/grid.yaml)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List experiments")
    list_p.add_argument("--only", help="Single experiment id")
    list_p.add_argument("--phase", help="Phase name or number")
    list_p.add_argument("--start-phase", type=int, default=1)
    list_p.add_argument("--max-phases", type=int, default=99)
    list_p.add_argument("-v", "--verbose", action="store_true")
    list_p.set_defaults(func=_cmd_list)

    val_p = sub.add_parser("validate", help="Validate grid.yaml")
    val_p.set_defaults(func=_cmd_validate)

    sft_p = sub.add_parser("sft", help="Run SFT stage from grid")
    sft_p.add_argument("--config", default="configs/qwen_4b_qlora.yaml")
    sft_p.add_argument("--module", default="rlvr_pipeline")
    sft_p.add_argument("--python", default=None)
    sft_p.add_argument("--sft-id", default="C2")
    sft_p.add_argument("--dry-run", action="store_true")
    sft_p.set_defaults(func=_cmd_sft)

    run_p = sub.add_parser("run", help="Run RL experiments from the grid")
    run_p.add_argument("--config", default="configs/qwen_4b_qlora.yaml")
    run_p.add_argument("--output", default="./runs/experiments")
    run_p.add_argument(
        "--sft-checkpoint",
        default="",
        help="Default SFT adapters (ignored when an experiment sets sft_checkpoint_path: '')",
    )
    run_p.add_argument(
        "--module",
        default="rlvr_pipeline",
        help="Train module (team_pack default: rlvr_pipeline)",
    )
    run_p.add_argument("--python", default=None, help="Python executable (default: current)")
    run_p.add_argument("--max-steps", type=int, default=300)
    run_p.add_argument("--wandb", action="store_true")
    run_p.add_argument("--wandb-group", default="arabic-rlvr-experiments")
    run_p.add_argument("--resume", action="store_true", help="Skip experiments with DONE marker / enough steps")
    run_p.add_argument("--dry-run", action="store_true", help="Print commands only")
    run_p.add_argument("--retries", type=int, default=1)
    run_p.add_argument("--only", help="Run a single experiment id (e.g. A1)")
    run_p.add_argument("--phase", help="Run one phase by name or number")
    run_p.add_argument("--start-phase", type=int, default=1)
    run_p.add_argument("--max-phases", type=int, default=99)
    run_p.add_argument("--run-sft-first", action="store_true")
    run_p.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
