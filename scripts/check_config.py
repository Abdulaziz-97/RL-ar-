"""
Quick Configuration Diagnostic Check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.trainer import build_trainer

def main():
    config_path = Path("configs/qwen_4b_2x5090_v3_sota.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} does not exist.")
        return 1

    cfg = RLVRConfig.from_yaml(config_path)
    print("================================================")
    print("RLVR PIPELINE STEP DIAGNOSTIC CHECK:")
    print(f"  * max_steps                   = {cfg.max_steps}")
    print(f"  * num_train_epochs            = {cfg.num_train_epochs}")
    print(f"  * gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    print(f"  * per_device_train_batch_size  = {cfg.per_device_train_batch_size}")
    print("=================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
