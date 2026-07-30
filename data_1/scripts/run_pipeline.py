"""CLI entry for this pack.

Runs the synth orchestrator with `backends/dspy_backend.py`:

  python scripts/run_pipeline.py --mode replay   # offline SHA match vs reference
  python scripts/run_pipeline.py --mode live     # live Pro/Flash generation (needs API key)

Replay clears `--work-dir` first so stage artifacts cannot mix with a prior run.
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
    parser.add_argument("--work-dir", type=Path, default=PACK_ROOT / "outputs" / "run")
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    args = parser.parse_args()

    os.environ["RLVR_PACK_MODE"] = args.mode
    from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator

    cfg = SynthConfig.from_yaml(args.config)
    cfg.work_dir = str(args.work_dir)
    cfg.backend = "external"
    cfg.external_module = str(PACK_ROOT / "backends" / "dspy_backend.py")
    cfg.external_config = {
        "mode": args.mode,
        "model": args.model,
        "gepa_path": str(PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"),
        "replay_stages": str(PACK_ROOT / "assets" / "replay_stages"),
        "budget_path": str(args.work_dir / "budget.json"),
        "budget_usd": 30.0,
        "cache": True,
    }
    cfg.partitions = {"sft_train": 1.0}
    cfg.max_alternate_methods = 0

    if args.work_dir.exists() and args.mode == "replay":
        shutil.rmtree(args.work_dir)

    result = SynthOrchestrator(cfg).run(resume=False)
    out = {
        "mode": args.mode,
        "result": result,
        "sft_train": str(Path(args.work_dir) / "release_corpora" / "sft_train.jsonl"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
