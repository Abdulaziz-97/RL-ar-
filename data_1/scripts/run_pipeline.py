"""CLI entry for this pack.

Runs the synth orchestrator with `backends/dspy_backend.py`:

  python scripts/run_pipeline.py --mode replay   # offline SHA match vs reference
  python scripts/run_pipeline.py --mode live     # live Pro/Flash generation (needs API key)

Replay clears `--work-dir` first so stage artifacts cannot mix with a prior run.
YAML `partitions` are respected (do not hardcode sft_train).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "src"))
sys.path.insert(0, str(PACK_ROOT / "vendor"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--config", type=Path, default=PACK_ROOT / "configs" / "pilot_20.yaml")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--track", choices=["auto", "sft", "rlvr"], default="auto",
                        help="Optional partition override; default keeps YAML partitions.")
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel teacher workers for SFT multi_trace (env TEACHER_WORKERS).",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    os.environ["RLVR_PACK_MODE"] = args.mode
    from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator

    cfg = SynthConfig.from_yaml(args.config)
    work_dir = args.work_dir or Path(cfg.work_dir)
    cfg.work_dir = str(work_dir)
    cfg.backend = "external"
    cfg.external_module = str(PACK_ROOT / "backends" / "dspy_backend.py")
    budget = float(args.budget_usd) if args.budget_usd is not None else 30.0
    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("TEACHER_WORKERS", "96" if args.track == "sft" else "1"))
    # RLVR is prompt-only; keep workers=1 to avoid pointless teacher spin-up fanout.
    if args.track == "rlvr":
        workers = 1
    cfg.external_config = {
        "mode": args.mode,
        "model": args.model,
        "gepa_path": str(PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"),
        "replay_stages": str(PACK_ROOT / "assets" / "replay_stages"),
        "budget_path": str(work_dir / "budget.json"),
        "budget_usd": budget,
        "cache": True,
        "teacher_workers": max(1, int(workers)),
        "teacher_retries": int(os.environ.get("TEACHER_RETRIES", "4")),
        "max_tokens": int(os.environ.get("TEACHER_MAX_TOKENS", "8192")),
    }
    cfg.max_alternate_methods = 0
    print(
        json.dumps(
            {
                "mode": args.mode,
                "track": args.track,
                "model": args.model,
                "teacher_workers": cfg.external_config["teacher_workers"],
                "budget_usd": budget,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.track == "sft":
        cfg.partitions = {"sft_train": 1.0}
    elif args.track == "rlvr":
        cfg.partitions = {"rlvr_train": 1.0}
        cfg.traces_per_problem = 0
    # else: keep YAML partitions verbatim

    if work_dir.exists() and (args.mode == "replay" or args.no_resume):
        shutil.rmtree(work_dir)

    result = SynthOrchestrator(cfg).run(resume=not args.no_resume and args.mode != "replay")
    corpora = work_dir / "release_corpora"
    out = {
        "mode": args.mode,
        "result": result,
        "partitions": cfg.partitions,
        "sft_train": str(corpora / "sft_train.jsonl"),
        "rlvr_train": str(corpora / "rlvr_train.jsonl"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
