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
    trainer = build_trainer(cfg)
    ds_len = len(trainer.train_dataset) if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None else "N/A"
    train_dl_len = len(trainer.get_train_dataloader()) if hasattr(trainer, "get_train_dataloader") else "N/A"
    inflation_factor = "N/A"
    if isinstance(ds_len, (int, float)) and isinstance(train_dl_len, (int, float)) and ds_len > 0:
        inflation_factor = f"{train_dl_len / ds_len:.2f}"

    if getattr(trainer, "lr_scheduler", None) is None and hasattr(trainer, "create_scheduler"):
        try:
            trainer.create_scheduler(num_training_steps=trainer.args.max_steps, optimizer=None)
        except Exception:
            pass

    scheduler_info = "N/A"
    if getattr(trainer, "lr_scheduler", None) is not None:
        sch = trainer.lr_scheduler
        if hasattr(sch, "state_dict") and callable(sch.state_dict):
            scheduler_info = sch.state_dict()
        else:
            scheduler_info = getattr(sch, "total_steps", "N/A")

    print("================================================")
    print("RLVR PIPELINE STEP DIAGNOSTIC CHECK:")
    print(f"  * raw_prompts (dataset)       = {ds_len}")
    print(f"  * len(train_dataloader)       = {train_dl_len}")
    print(f"  * dataloader inflation_factor = {inflation_factor}")
    print(f"  * max_steps                   = {trainer.args.max_steps}")
    print(f"  * num_train_epochs            = {trainer.args.num_train_epochs}")
    print(f"  * gradient_accumulation_steps = {trainer.args.gradient_accumulation_steps}")
    print(f"  * per_device_train_batch_size  = {trainer.args.per_device_train_batch_size}")
    print(f"  * scheduler info              = {scheduler_info}")
    print("=================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
