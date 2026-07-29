"""
Automatic Benchmark Telemetry Probe Callback for HuggingFace Trainer.

Fires on_save at every save_steps (e.g. step 50, 100, 150, 200).
Evaluates AraMath (full), AraPro (fixed 500 sample), AraTruthfulQA (full) natively,
logs accuracies to W&B under probe/aramath, probe/arapro, probe/truthfulqa.
"""

from __future__ import annotations

import os
from pathlib import Path
from transformers.trainer_callback import TrainerCallback


class BenchmarkProbeCallback(TrainerCallback):
    """Automatically runs fast benchmark probe when checkpoints are saved."""

    def __init__(self, base_model: str = "unsloth/Qwen3.5-4B", enable_probe: bool = True):
        self.base_model = base_model
        self.enable_probe = enable_probe

    def on_save(self, args, state, control, **kwargs):
        if not self.enable_probe:
            return
        # Run probe on main process (rank 0) only
        if getattr(args, "process_index", 0) != 0:
            return

        step = state.global_step
        output_dir = getattr(args, "output_dir", "./outputs")
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        if not os.path.exists(ckpt_dir):
            ckpt_dir = output_dir

        print(f"\n[AUTO-PROBE] Step {step}: Running benchmark probe on {ckpt_dir}...", flush=True)
        try:
            from scripts.run_benchmark_probe import run_probe
            metrics = run_probe(ckpt_dir, base_model=self.base_model)
            if getattr(args, "report_to", None) and "wandb" in args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        log_dict = {f"probe/{k}": v for k, v in metrics.items()}
                        wandb.log(log_dict, step=step)
                except Exception:
                    pass
        except Exception as e:
            print(f"[AUTO-PROBE] Warning: Probe skipped due to error: {e}", flush=True)
