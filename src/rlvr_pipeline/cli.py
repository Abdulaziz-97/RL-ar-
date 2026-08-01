"""
Plug-and-play CLI.

    python -m rlvr_pipeline sft   --config configs/qwen_4b_qlora.yaml
    python -m rlvr_pipeline train --config configs/qwen_4b_smoke_v11.yaml --sft-checkpoint ./runs/sft_v1
    python -m rlvr_pipeline eval  --config configs/qwen_4b_qlora.yaml --data data/... --checkpoint outputs/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.trainer import build_trainer, build_sft_trainer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlvr_pipeline",
        description="Arabic Reasoning RLVR Pipeline (TRL GRPOTrainer + QLoRA)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── SFT ──
    sft_parser = subparsers.add_parser("sft", help="Stage 1: Cold-start SFT on CoT solutions")
    sft_parser.add_argument("--config", required=True, help="Path to YAML config file")
    sft_parser.add_argument("--data", help="Override SFT data path (JSONL)")
    sft_parser.add_argument("--model", help="Override model name/path")
    sft_parser.add_argument("--output", help="Override output directory")
    sft_parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    sft_parser.add_argument("--max-steps", type=int, help="Override max steps")
    sft_parser.add_argument("--num-train-epochs", type=int, dest="num_train_epochs", help="Override epochs")
    sft_parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    sft_parser.add_argument("--per-device-batch-size", type=int, dest="per_device_train_batch_size",
                            help="Override batch size")
    sft_parser.add_argument("--grad-accum", type=int, dest="gradient_accumulation_steps",
                            help="Override gradient accumulation steps")

    # ── Train (GRPO) ──
    train_parser = subparsers.add_parser("train", help="Stage 2: GRPO RLVR training")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument("--data", help="Override train data path (JSONL)")
    train_parser.add_argument("--eval", help="Eval data path (JSONL)")
    train_parser.add_argument("--model", help="Override model name/path")
    train_parser.add_argument("--output", help="Override output directory")
    train_parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    train_parser.add_argument("--wandb-group", help="wandb group name for experiment grouping")
    train_parser.add_argument("--max-steps", type=int, help="Override max steps")

    # Algorithm flags
    train_parser.add_argument("--loss-type", choices=["grpo", "dapo", "dr_grpo", "sapo", "bnpo"],
                              help="Override loss type")
    train_parser.add_argument("--importance-sampling", dest="importance_sampling_level",
                              choices=["token", "sequence", "sequence_token"],
                              help="Override importance sampling")
    train_parser.add_argument("--scale-rewards", choices=["group", "batch", "off"],
                              help="Override reward scaling")
    train_parser.add_argument("--curriculum", dest="curriculum_schedule_type",
                              choices=["gaussian", "fixed_switch", "random_mix", "none"],
                              help="Override curriculum schedule")
    train_parser.add_argument("--zero-variance", dest="zero_variance_strategy",
                              choices=["direct_scoring", "discard"],
                              help="Override zero-variance strategy")
    train_parser.add_argument("--enable-crps", dest="enable_crps", action="store_true", default=None,
                              help="Enable CRPS progressive suffix hints")
    train_parser.add_argument("--disable-crps", dest="enable_crps", action="store_false", default=None,
                              help="Disable CRPS")
    train_parser.add_argument("--sft-checkpoint", dest="sft_checkpoint_path",
                              help="Path to LoRA adapters from SFT stage")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume GRPO from the latest checkpoint under output_dir (opt-in)",
    )

    # Hyperparameter overrides
    train_parser.add_argument("--temperature", type=float, help="Override generation temperature")
    train_parser.add_argument("--top-p", type=float, dest="top_p", help="Override top_p")
    train_parser.add_argument("--top-k", type=int, dest="top_k", help="Override top_k")
    train_parser.add_argument("--num-generations", type=int, dest="num_generations",
                              help="Override num_generations")
    train_parser.add_argument("--max-completion-length", type=int, dest="max_completion_length",
                              help="Override max completion length")
    train_parser.add_argument("--sigma-fraction", type=float, dest="sigma_fraction",
                              help="Override curriculum sigma")
    train_parser.add_argument("--beta", type=float, help="Override KL penalty beta")
    train_parser.add_argument("--epsilon", type=float, help="Override PPO clip low")
    train_parser.add_argument("--epsilon-high", type=float, dest="epsilon_high",
                              help="Override PPO clip high")
    train_parser.add_argument("--learning-rate", type=float, dest="learning_rate",
                              help="Override learning rate")
    train_parser.add_argument("--per-device-batch-size", type=int, dest="per_device_train_batch_size",
                              help="Override batch size")
    train_parser.add_argument("--grad-accum", type=int, dest="gradient_accumulation_steps",
                              help="Override gradient accumulation steps")

    # ── Eval ──
    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument("--config", required=True, help="Path to YAML config file")
    eval_parser.add_argument("--data", required=True, help="Eval data path (JSONL)")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")

    # ── Audit cold-start ──
    audit_parser = subparsers.add_parser(
        "audit-coldstart", help="Audit cold-start JSONL for tag-boundary purity"
    )
    audit_parser.add_argument(
        "--data",
        default="data/arabic_reasoning_coldstart.jsonl",
        help="Cold-start JSONL path",
    )
    audit_parser.add_argument("--max-preamble-words", type=int, default=5)
    audit_parser.add_argument("--min-quality", type=float, default=0.5)
    audit_parser.add_argument("--min-arabic-purity", type=float, default=0.7)
    audit_parser.add_argument("--top", type=int, default=20)
    audit_parser.add_argument("--json-out", help="Write full JSON report")
    audit_parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="Exit 1 if impure_rate exceeds this fraction",
    )

    return parser


def _apply_overrides(config: RLVRConfig, args: argparse.Namespace) -> RLVRConfig:
    overrides = [
        ("train_data_path", "data"),
        ("eval_data_path", "eval"),
        ("model_name", "model"),
        ("output_dir", "output"),
        ("max_steps", "max_steps"),
        ("num_train_epochs", "num_train_epochs"),
        ("learning_rate", "learning_rate"),
        ("loss_type", "loss_type"),
        ("importance_sampling_level", "importance_sampling_level"),
        ("scale_rewards", "scale_rewards"),
        ("curriculum_schedule_type", "curriculum_schedule_type"),
        ("zero_variance_strategy", "zero_variance_strategy"),
        ("sft_checkpoint_path", "sft_checkpoint_path"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("num_generations", "num_generations"),
        ("max_completion_length", "max_completion_length"),
        ("sigma_fraction", "sigma_fraction"),
        ("beta", "beta"),
        ("epsilon", "epsilon"),
        ("epsilon_high", "epsilon_high"),
        ("per_device_train_batch_size", "per_device_train_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ]
    for config_attr, cli_attr in overrides:
        val = getattr(args, cli_attr, None)
        if val is not None:
            setattr(config, config_attr, val)

    if hasattr(args, "enable_crps") and args.enable_crps is not None:
        config.enable_crps = args.enable_crps

    if hasattr(args, "wandb") and args.wandb:
        config.use_wandb = True
        config.report_to = "wandb"

    if hasattr(args, "wandb_group") and args.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "audit-coldstart":
        from rlvr_pipeline.audit_coldstart import main as audit_main

        audit_argv = ["--data", args.data]
        audit_argv += ["--max-preamble-words", str(args.max_preamble_words)]
        audit_argv += ["--min-quality", str(args.min_quality)]
        audit_argv += ["--min-arabic-purity", str(args.min_arabic_purity)]
        audit_argv += ["--top", str(args.top)]
        if args.json_out:
            audit_argv += ["--json-out", args.json_out]
        if args.fail_above is not None:
            audit_argv += ["--fail-above", str(args.fail_above)]
        return audit_main(audit_argv)

    config = RLVRConfig.from_yaml(args.config)
    config = _apply_overrides(config, args)

    if args.command == "sft":
        if hasattr(args, "data") and args.data:
            config.coldstart_data_path = args.data
        if not config.coldstart_data_path:
            print("Error: --data or config.coldstart_data_path is required", file=sys.stderr)
            return 1
        # SFT CLI --max-steps must set sft_max_steps, never GRPO max_steps.
        if getattr(args, "max_steps", None) is not None:
            config.sft_max_steps = args.max_steps
            config.max_steps = None
        if getattr(args, "num_train_epochs", None) is not None:
            config.sft_num_train_epochs = args.num_train_epochs
        if getattr(args, "learning_rate", None) is not None:
            config.sft_learning_rate = args.learning_rate
        if getattr(args, "per_device_train_batch_size", None) is not None:
            config.sft_per_device_train_batch_size = args.per_device_train_batch_size
        if getattr(args, "gradient_accumulation_steps", None) is not None:
            config.sft_gradient_accumulation_steps = args.gradient_accumulation_steps

        print(f"Stage 1: Cold-start SFT")
        print(f"Model:    {config.model_name}")
        print(f"Data:     {config.coldstart_data_path}")
        print(f"Output:   {config.output_dir}")
        print(f"4-bit:    {config.load_in_4bit}")
        print()

        trainer = build_sft_trainer(config)
        trainer.train()
        trainer.save_model()
        try:
            from rlvr_pipeline.lineage import LineageMeta, write_lineage

            write_lineage(
                config.output_dir,
                LineageMeta(
                    base_model=config.model_name,
                    instruction_adapter=config.instruction_base_model,
                    stage="sft",
                    adapter_path=config.output_dir,
                    config_hash=None,
                    dataset_hash=None,
                ),
            )
        except Exception as e:
            print(f"Warning: failed to write lineage.json: {e}", flush=True)
        print(f"Adapters saved to {config.output_dir}")
        return 0

    elif args.command == "train":
        if not config.train_data_path:
            print("Error: --data or config.train_data_path is required", file=sys.stderr)
            return 1

        print(f"Stage 2: GRPO RLVR Training")
        print(f"Model:    {config.model_name}")
        print(f"Data:     {config.train_data_path}")
        print(f"Output:   {config.output_dir}")
        print(f"Loss:     {config.loss_type}")
        print(f"IS:       {config.importance_sampling_level}")
        print(f"Scale:    {config.scale_rewards}")
        print(f"Curric.:  {config.curriculum_schedule_type}")
        print(f"Z-var:    {config.zero_variance_strategy}")
        print(f"CRPS:     {config.enable_crps}")
        print(f"SFT-ckpt: {config.sft_checkpoint_path or '(none)'}")
        print(f"G:        {config.num_generations}")
        print(f"MaxLen:   {config.max_completion_length}")
        print(f"Beta:     {config.beta}  Eps: {config.epsilon}/{config.epsilon_high}")
        print(f"Temp:     {config.temperature}  TopP: {config.top_p}  TopK: {config.top_k}")
        print(f"LR:       {config.learning_rate}")
        print(f"Weights:  {config.reward_weights}")
        print()

        try:
            from rlvr_pipeline.rewards import diagnose_reward_weights

            diagnose_reward_weights(config.reward_weights)
        except Exception as e:
            print(f"Reward diagnostics failed: {e}", file=sys.stderr)
            return 1

        trainer = build_trainer(config)
        ds_len = len(trainer.train_dataset) if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None else "N/A"
        train_dl_len = len(trainer.get_train_dataloader()) if hasattr(trainer, "get_train_dataloader") else "N/A"
        inflation_factor = "N/A"
        if isinstance(ds_len, (int, float)) and isinstance(train_dl_len, (int, float)) and ds_len > 0:
            inflation_factor = f"{train_dl_len / ds_len:.2f}"

        scheduler_info = "N/A"
        if getattr(trainer, "lr_scheduler", None) is not None:
            sch = trainer.lr_scheduler
            if hasattr(sch, "state_dict") and callable(sch.state_dict):
                scheduler_info = sch.state_dict()
            else:
                scheduler_info = getattr(sch, "total_steps", "N/A")

        print("=================================================================")
        print("VERIFIED TRAINER STEP CONFIGURATION:")
        print(f"  * raw_prompts (dataset)                       = {ds_len}")
        print(f"  * len(train_dataloader)                       = {train_dl_len}")
        print(f"  * dataloader inflation_factor                 = {inflation_factor}")
        print(f"  * trainer.args.max_steps                      = {getattr(trainer.args, 'max_steps', 'N/A')}")
        print(f"  * trainer.args.num_train_epochs               = {getattr(trainer.args, 'num_train_epochs', 'N/A')}")
        print(f"  * trainer.args.gradient_accumulation_steps    = {getattr(trainer.args, 'gradient_accumulation_steps', 'N/A')}")
        print(f"  * trainer.args.per_device_train_batch_size     = {getattr(trainer.args, 'per_device_train_batch_size', 'N/A')}")
        print(f"  * scheduler info                              = {scheduler_info}")
        print("=================================================================", flush=True)

        resume_requested = bool(getattr(args, "resume", False)) or (
            getattr(config, "resume_policy", "fresh") == "resume"
        )
        if getattr(config, "resume_policy", "fresh") == "fail-if-output-exists":
            out = Path(config.output_dir)
            if out.exists() and any(out.glob("checkpoint-*")):
                print(
                    f"Error: resume_policy=fail-if-output-exists and checkpoints exist in {out}",
                    file=sys.stderr,
                )
                return 1

        latest_checkpoint = None
        if resume_requested:
            sdir = config.output_dir
            if sdir and os.path.exists(sdir):
                ckpts = [
                    os.path.join(sdir, d)
                    for d in os.listdir(sdir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(sdir, d))
                ]
                if ckpts:
                    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
                    latest_checkpoint = ckpts[-1]

        if latest_checkpoint:
            print(f"Resuming GRPO training from checkpoint: {latest_checkpoint}", flush=True)
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            if resume_requested:
                print("Resume requested but no checkpoint found; starting fresh.", flush=True)
            else:
                print(
                    f"Fresh run (resume_policy={getattr(config, 'resume_policy', 'fresh')}); "
                    "ignoring any existing checkpoints.",
                    flush=True,
                )
            trainer.train()
        trainer.save_model()
        try:
            from rlvr_pipeline.lineage import LineageMeta, write_lineage

            write_lineage(
                config.output_dir,
                LineageMeta(
                    base_model=config.model_name,
                    instruction_adapter=config.instruction_base_model,
                    sft_parent=config.sft_checkpoint_path or None,
                    stage="grpo",
                    adapter_path=config.output_dir,
                ),
            )
        except Exception as e:
            print(f"Warning: failed to write lineage.json: {e}", flush=True)
        return 0

    elif args.command == "eval":
        config.eval_data_path = args.data
        # Trainer checkpoints are PEFT adapters, not standalone base models.
        # Keep config.model_name as the base and load the adapter strictly.
        config.sft_checkpoint_path = args.checkpoint
        trainer = build_trainer(config, evaluation_only=True)
        metrics = trainer.evaluate()
        print(f"Eval metrics: {metrics}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
