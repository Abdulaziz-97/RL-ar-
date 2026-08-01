"""Checkpoint integrity checks for adapter-only V4 saves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FORBIDDEN_FULL_WEIGHT_GLOBS = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "model-*-of-*.safetensors",
)


def verify_adapter_only(checkpoint: Path) -> list[str]:
    errors: list[str] = []
    if not (checkpoint / "adapter_config.json").is_file():
        errors.append("missing adapter_config.json")
    has_weights = (checkpoint / "adapter_model.safetensors").is_file() or (
        checkpoint / "adapter_model.bin"
    ).is_file()
    if not has_weights:
        errors.append("missing adapter_model.safetensors/.bin")

    for pattern in FORBIDDEN_FULL_WEIGHT_GLOBS:
        hits = list(checkpoint.glob(pattern))
        # PEFT sometimes writes adapter_model.safetensors — allow that name only.
        hits = [h for h in hits if h.name not in {"adapter_model.safetensors", "adapter_model.bin"}]
        if hits:
            errors.append(f"full-model weight files present (not adapter-only): {[h.name for h in hits]}")
    return errors


def verify_lineage(checkpoint: Path) -> list[str]:
    errors: list[str] = []
    lineage = checkpoint / "lineage.json"
    if not lineage.is_file():
        # Soft warning for older checkpoints; treat as error only if required later.
        return errors
    try:
        data = json.loads(lineage.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"invalid lineage.json: {e}"]
    if not data.get("base_model"):
        errors.append("lineage.json missing base_model")
    return errors


def verify_tensors_finite(checkpoint: Path) -> list[str]:
    errors: list[str] = []
    weight = checkpoint / "adapter_model.safetensors"
    if not weight.is_file():
        return errors
    try:
        from safetensors import safe_open
        import math

        with safe_open(str(weight), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if not bool(t.isfinite().all()):
                    errors.append(f"non-finite tensors in adapter key {key}")
                    break
                # Huge embedding tensors should not appear for V4 targets.
                if "embed_tokens" in key or "lm_head" in key:
                    if t.numel() > 1_000_000:
                        errors.append(
                            f"unexpected large embedding adapter tensor: {key} shape={tuple(t.shape)}"
                        )
    except Exception as e:
        errors.append(f"failed reading adapter tensors: {e}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify RLVR checkpoint integrity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--require-adapter-only", action="store_true")
    parser.add_argument("--require-lineage", action="store_true")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint path does not exist: {ckpt}", file=sys.stderr)
        return 1

    errors: list[str] = []
    if args.require_adapter_only:
        errors.extend(verify_adapter_only(ckpt))
    if args.require_lineage:
        lin_errs = verify_lineage(ckpt)
        if not (ckpt / "lineage.json").is_file():
            errors.append("missing lineage.json")
        errors.extend(lin_errs)
    else:
        errors.extend(verify_lineage(ckpt))
    errors.extend(verify_tensors_finite(ckpt))

    if errors:
        print("CHECKPOINT INTEGRITY FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"CHECKPOINT INTEGRITY OK: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
