"""Preflight checks for V4 production training runs."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import shutil
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "transformers": "5.13.0",
    "trl": "1.7.1",
    "peft": "0.19.1",
    "PyYAML": "6.0.2",
}


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check_versions() -> list[str]:
    errors: list[str] = []
    for name, expected in REQUIRED_PACKAGES.items():
        got = _pkg_version(name)
        if got is None:
            errors.append(f"Missing package: {name} (expected {expected})")
        elif got != expected:
            errors.append(f"Version mismatch: {name}={got} (expected {expected})")
    # Training path uses HF generate (use_vllm=false). Installed vLLM 0.26 is a
    # soft warning only — do not hard-fail overnight GRPO/SFT runs.
    vllm = _pkg_version("vllm")
    if vllm and vllm.startswith("0.26"):
        print(
            f"WARNING: Unsupported vLLM {vllm} with locked TRL; "
            "ignored for HF training (use_vllm=false). Uninstall for eval/vLLM probes.",
            file=sys.stderr,
            flush=True,
        )
    return errors


def check_resources(require_gpus: int | None, min_disk_gb: float = 20.0) -> list[str]:
    errors: list[str] = []
    free = shutil.disk_usage(".").free / (1024**3)
    if free < min_disk_gb:
        errors.append(f"Insufficient disk free space: {free:.1f}GB < {min_disk_gb}GB")

    if require_gpus is not None and require_gpus > 0:
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("CUDA is not available but GPUs were required")
            else:
                n = torch.cuda.device_count()
                if n < require_gpus:
                    errors.append(f"Need {require_gpus} GPUs, found {n}")
                # bf16 support probe on first device
                major, _ = torch.cuda.get_device_capability(0)
                if major < 8:
                    errors.append(
                        f"GPU compute capability {major}.x may lack reliable bf16"
                    )
        except Exception as e:
            errors.append(f"CUDA probe failed: {e}")
    return errors


def check_datasets(config_path: Path) -> list[str]:
    from rlvr_pipeline.config import RLVRConfig

    errors: list[str] = []
    cfg = RLVRConfig.from_yaml(config_path)
    for label, path in (
        ("train", cfg.train_data_path),
        ("coldstart", cfg.coldstart_data_path),
    ):
        if not path:
            errors.append(f"Missing {label}_data_path in config")
            continue
        p = Path(path)
        if not p.is_file():
            errors.append(f"Missing {label} dataset file: {p}")
    return errors


def check_output_state(path: str | None, policy: str = "fresh") -> list[str]:
    errors: list[str] = []
    if not path:
        return errors
    out = Path(path)
    if policy == "fail-if-output-exists" and out.exists() and any(out.glob("checkpoint-*")):
        errors.append(f"Output already has checkpoints under {out}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V4 production preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-gpus", type=int, default=None)
    parser.add_argument("--sft-output", default=None)
    parser.add_argument("--grpo-output", default=None)
    parser.add_argument("--min-disk-gb", type=float, default=20.0)
    args = parser.parse_args(argv)

    errors: list[str] = []
    errors.extend(check_versions())
    errors.extend(check_resources(args.require_gpus, args.min_disk_gb))
    errors.extend(check_datasets(Path(args.config)))
    from rlvr_pipeline.config import RLVRConfig

    cfg = RLVRConfig.from_yaml(args.config)
    policy = getattr(cfg, "resume_policy", "fresh")
    errors.extend(check_output_state(args.sft_output, policy))
    errors.extend(check_output_state(args.grpo_output, policy))

    if errors:
        print("PREFLIGHT FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("PREFLIGHT OK")
    print(f"  config={args.config}")
    print(f"  resume_policy={policy}")
    print(f"  require_gpus={args.require_gpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
