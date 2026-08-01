"""
Automatic Benchmark Telemetry Probe Callback for HuggingFace Trainer.

Fires on_save at every save_steps when enable_probe=True.
Prefer running probes as a separate post-save stage with released GPUs.
"""

from __future__ import annotations

import os

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
            import subprocess
            import sys

            # Strip DDP env vars so vLLM initializes cleanly in subprocess
            clean_env = os.environ.copy()
            for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]:
                clean_env.pop(key, None)

            cmd = [
                sys.executable,
                "scripts/run_benchmark_probe.py",
                "--checkpoint",
                ckpt_dir,
                "--base-model",
                self.base_model,
            ]
            res = subprocess.run(cmd, env=clean_env, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                print(f"[AUTO-PROBE] Probe Output:\n{res.stdout}", flush=True)
            else:
                print(
                    f"[AUTO-PROBE] Warning: Probe subprocess returned non-zero code "
                    f"{res.returncode}:\n{res.stderr}",
                    flush=True,
                )
        except Exception as e:
            print(f"[AUTO-PROBE] Warning: Probe skipped due to error: {e}", flush=True)
