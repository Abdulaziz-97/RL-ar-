"""Trainer assembly: SFT cold-start + GRPO with QLoRA, Arabic rewards, curriculum, CRPS."""

from __future__ import annotations

from typing import Any, Optional

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.data import load_rlvr_dataset, load_cold_start_sft_dataset
from rlvr_pipeline.rewards import ALL_REWARD_FUNCS, DEFAULT_REWARD_WEIGHTS

import os
import sys
import re
from pathlib import Path

# Qwen3.5 injects a pre-closed empty <think></think> into generation prompts.
# That makes reward_format unearnable; strip it so rollouts match SFT targets.
_EMPTY_THINK_INJECTION = "{{- '<think>\\n\\n</think>\\n\\n' }}"


def strip_think_injection(template: str | None) -> str | None:
    """Remove the pre-closed empty <think> block from a chat template string."""
    if not template:
        return template
    # Match Jinja tags containing pre-closed empty <think>...</think>
    cleaned = re.sub(
        r"\{\{-\s*['\"]<think>(?:(?!<think>).)*?</think>.*?['\"]\s*\}\}",
        "",
        template,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"<think>\s*</think>\s*", "", cleaned)
    return cleaned





def load_sft_adapter_strict(peft_model, ckpt_dir: str) -> None:
    """Load SFT LoRA into an existing PEFT model, remapping Qwen3.5 key prefixes.

    Checkpoints may use ``model.language_model.layers.*`` while GRPO loads
    ``model.layers.*``. PEFT warns and silently loads nothing on mismatch;
    this remaps keys and fails hard if anything is left unmatched.
    """
    from pathlib import Path

    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    ckpt = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    live_keys = set(get_peft_model_state_dict(peft_model).keys())

    remapped = {}
    for key, value in ckpt.items():
        candidates = [
            key,
            key.replace(".language_model.", "."),
            key.replace(".model.model.", ".model."),
            key.replace(".model.language_model.", ".model."),
        ]
        if key.startswith("base_model.model.model."):
            candidates.append("base_model.model." + key[len("base_model.model.model."):])
        elif key.startswith("base_model.model."):
            candidates.append("base_model.model.model." + key[len("base_model.model."):])

        target = next((c for c in candidates if c in live_keys), None)
        if target is None and ".layers." in key:
            suffix = key.split(".layers.", 1)[-1]
            target = next((lk for lk in live_keys if lk.endswith(suffix)), None)

        if target is None:
            raise RuntimeError(f"SFT adapter key cannot be mapped onto live model: {key}")
        remapped[target] = value

    missing = live_keys - set(remapped)
    if missing:
        raise RuntimeError(
            f"SFT adapter checkpoint is missing {len(missing)} keys, e.g. {sorted(missing)[:3]}"
        )

    set_peft_model_state_dict(peft_model, remapped)
    print(f"SFT adapter loaded strictly: {len(remapped)} tensors from {ckpt_dir}", flush=True)


def fix_chat_template(processing_class) -> bool:
    """Patch a tokenizer/processor in place. Returns True if a fix was applied."""
    if processing_class is None:
        return False
    tok = processing_class
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    template = getattr(tok, "chat_template", None)
    fixed = strip_think_injection(template)
    if fixed == template:
        return False
    tok.chat_template = fixed
    if tok is not processing_class and hasattr(processing_class, "chat_template"):
        processing_class.chat_template = fixed
    return True


def _attach_stop_string_criteria(trainer, stop_strings: list[str]) -> None:
    """Stop at </answer> via StopStringCriteria (TRL generate has no tokenizer=)."""
    if not stop_strings:
        return

    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        StopStringCriteria,
    )

    tok = getattr(trainer, "processing_class", None) or getattr(trainer, "_tokenizer", None)
    if tok is None:
        return
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    criteria = StoppingCriteriaList(
        [StopStringCriteria(tokenizer=tok, stop_strings=stop_strings)]
    )

    gen_config = getattr(trainer, "generation_config", None)
    if gen_config is not None and getattr(gen_config, "stop_strings", None):
        gen_config.stop_strings = None

    targets = [trainer.model]
    base_getter = getattr(trainer.model, "get_base_model", None)
    if callable(base_getter):
        try:
            base = base_getter()
            if base is not None and base not in targets:
                targets.append(base)
        except Exception:
            pass

    for model in targets:
        if getattr(model, "_rlvr_answer_stop_patched", False):
            continue
        original_generate = model.generate

        def generate_with_answer_stop(*args, _original=original_generate, **kwargs):
            kwargs.pop("mm_token_type_ids", None)
            existing = kwargs.get("stopping_criteria")
            if existing is None:
                kwargs["stopping_criteria"] = criteria
            else:
                kwargs["stopping_criteria"] = StoppingCriteriaList(
                    list(existing) + list(criteria)
                )
            gc = kwargs.get("generation_config")
            if gc is not None and getattr(gc, "stop_strings", None):
                gc.stop_strings = None
            return _original(*args, **kwargs)

        model.generate = generate_with_answer_stop
        model._rlvr_answer_stop_patched = True


def _align_trainable_dtype_for_amp(trainer, *, fp16: bool, bf16: bool) -> None:
    """Make QLoRA trainable weights GradScaler-safe on RTX 20-series.

    Qwen3.5 adapters often inherit bf16. With ``fp16=True`` GradScaler then either:
      - fails on bf16 grads (no CUDA unscale kernel), or
      - fails after casting adapters to fp16 ("Attempting to unscale FP16 gradients").
    Keep LoRA in float32 and disable AMP for this path.
    """
    model = getattr(trainer, "model", None)
    if model is None or bf16:
        return
    if not fp16:
        return

    import torch

    cast = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)
            cast += 1

    args = getattr(trainer, "args", None)
    if args is not None:
        args.fp16 = False
        args.bf16 = False

    print(
        f"QLoRA AMP fix: cast {cast} trainable params to float32; disabled fp16/bf16 GradScaler",
        flush=True,
    )


def build_sft_trainer(
    config: RLVRConfig,
    sft_dataset=None,
    model=None,
    processing_class=None,
):
    """Build a TRL SFTTrainer for cold-start CoT distillation.

    Sets a minimal chat template so Qwen3.5's thinking-mode tokens don't
    override the <think>/<answer> XML format we're teaching.
    """
    from trl import SFTConfig, SFTTrainer

    if sft_dataset is None:
        if not config.coldstart_data_path:
            raise ValueError("config.coldstart_data_path must be set for SFT stage")
        sft_dataset = load_cold_start_sft_dataset(
            config.coldstart_data_path,
            system_prompt=config.system_prompt,
        )

    peft_config = config.build_peft_config() if model is None else None
    model_init_kwargs = config.build_model_init_kwargs() if model is None else None

    sft_lr = getattr(config, "sft_learning_rate", 2.0e-4)
    sft_epochs = getattr(config, "sft_num_train_epochs", None)
    if sft_epochs is None:
        sft_epochs = config.num_train_epochs
    sft_bs = getattr(config, "sft_per_device_train_batch_size", 4)
    sft_accum = getattr(config, "sft_gradient_accumulation_steps", 1)
    config.sync_wandb_env()
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        learning_rate=sft_lr,
        num_train_epochs=sft_epochs,
        per_device_train_batch_size=sft_bs,
        gradient_accumulation_steps=sft_accum,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=config.weight_decay,
        optim=config.optim,
        seed=config.seed,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        dataset_num_proc=8,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to if config.use_wandb else "none",
        loss_type="nll",
    )
    # Never inherit GRPO max_steps into SFT — use dedicated sft_max_steps only.
    sft_max_steps = getattr(config, "sft_max_steps", None)
    if sft_max_steps is not None and sft_max_steps > 0:
        sft_config.max_steps = int(sft_max_steps)
    if model_init_kwargs:
        sft_config.model_init_kwargs = model_init_kwargs

    if model is None:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel, get_peft_model
        init_kwargs = config.build_model_init_kwargs()
        base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **init_kwargs)
        if not hasattr(base_model.config, "text_config"):
            base_model.config.text_config = base_model.config

        if config.instruction_base_model:
            print(f"Merging SOTA instruction base adapter ({config.instruction_base_model}) into base weights...", flush=True)
            t06_peft = PeftModel.from_pretrained(base_model, config.instruction_base_model)
            base_model = t06_peft.merge_and_unload()
            print(f"Initializing fresh Rank-{config.lora_r} (alpha={config.lora_alpha}) LoRA adapter for SFT warm-up...", flush=True)
            peft_config = config.build_peft_config()
            model = get_peft_model(base_model, peft_config)
            peft_config = None
        else:
            model = get_peft_model(base_model, peft_config)
            peft_config = None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )

    if fix_chat_template(getattr(trainer, "processing_class", None)):
        print("Chat template fixed: removed empty <think> injection (SFT)", flush=True)

    _align_trainable_dtype_for_amp(trainer, fp16=config.fp16, bf16=config.bf16)
    _maybe_attach_hub_checkpoint_callback(trainer, config)
    return trainer


def _maybe_attach_hub_checkpoint_callback(trainer, config: RLVRConfig) -> None:
    """Attach HubCheckpointCallback when push_checkpoints_to_hub is enabled."""
    if not bool(getattr(config, "push_checkpoints_to_hub", False)):
        return
    from rlvr_pipeline.hub_checkpoint_callback import HubCheckpointCallback

    hub_id = getattr(config, "hub_model_id", None) or ""
    keep_n = getattr(config, "hub_keep_local_last_n", None)
    if keep_n is None:
        keep_n = 1
    name_prefix = getattr(config, "hub_checkpoint_name_prefix", None) or "checkpoint-evaluated"
    cb = HubCheckpointCallback(
        hub_model_id=hub_id,
        enabled=True,
        hub_private=bool(getattr(config, "hub_private", True)),
        delete_local_after_push=bool(
            getattr(config, "delete_local_checkpoint_after_hub_push", True)
        ),
        keep_local_last_n=int(keep_n),
        hub_checkpoint_name_prefix=str(name_prefix),
    )
    trainer.add_callback(cb)
    print(
        f"Hub checkpoint push ENABLED → {hub_id or '(missing hub_model_id)'} "
        f"as {name_prefix}-{{step}} "
        f"(private={cb.hub_private}, delete_local_after_push={cb.delete_local_after_push}, "
        f"keep_local_last_n={keep_n}).",
        flush=True,
    )


def build_trainer(
    config: RLVRConfig,
    train_dataset=None,
    eval_dataset=None,
    reward_funcs=None,
    reward_weights=None,
    model=None,
    processing_class=None,
    evaluation_only: bool = False,
):
    """Build GRPOTrainer (Stage 2). Run build_sft_trainer() first for cold-start."""
    if reward_funcs is None:
        reward_funcs = ALL_REWARD_FUNCS
    if reward_weights is None:
        reward_weights = config.reward_weights if config.reward_weights else DEFAULT_REWARD_WEIGHTS
    config.reward_weights = reward_weights

    if train_dataset is None and not evaluation_only:
        if not config.train_data_path:
            raise ValueError("Either train_dataset or config.train_data_path must be set")
        print("Loading training data...", flush=True)
        train_dataset = load_rlvr_dataset(
            config.train_data_path,
            system_prompt=config.system_prompt,
        )
        print(f"Loaded {len(train_dataset)} training samples", flush=True)

    if eval_dataset is None and config.eval_data_path:
        full_eval_ds = load_rlvr_dataset(
            config.eval_data_path,
            system_prompt=config.system_prompt,
        )
        max_eval_samples = getattr(config, "max_eval_samples", 32)
        if max_eval_samples and len(full_eval_ds) > max_eval_samples:
            print(f"Limiting eval dataset from {len(full_eval_ds)} to {max_eval_samples} samples for fast mid-run evaluation (~15 min)...", flush=True)
            eval_dataset = full_eval_ds.select(range(max_eval_samples))
        else:
            eval_dataset = full_eval_ds

    print("Building GRPO config...", flush=True)

    peft_config = None
    if model is None:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel, get_peft_model
        from rlvr_pipeline.lineage import read_lineage

        init_kwargs = config.build_model_init_kwargs()
        base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **init_kwargs)
        if not hasattr(base_model.config, "text_config"):
            base_model.config.text_config = base_model.config

        lineage_instruction = None
        if config.sft_checkpoint_path and os.path.isdir(config.sft_checkpoint_path):
            meta = read_lineage(config.sft_checkpoint_path)
            if meta and meta.instruction_adapter:
                lineage_instruction = str(meta.instruction_adapter).strip() or None

        cfg_instruction = (config.instruction_base_model or "").strip() or None
        if (
            lineage_instruction
            and cfg_instruction
            and cfg_instruction != lineage_instruction
        ):
            raise ValueError(
                "Fail-closed GRPO build: instruction_base_model mismatch vs SFT lineage "
                f"(config={cfg_instruction!r}, lineage={lineage_instruction!r})"
            )
        required_instruction = cfg_instruction or lineage_instruction

        # Always merge instruction base when config or SFT lineage requires it.
        if required_instruction:
            print(
                f"Merging SOTA instruction base adapter ({required_instruction}) "
                "into base weights...",
                flush=True,
            )
            t06_peft = PeftModel.from_pretrained(base_model, required_instruction)
            base_model = t06_peft.merge_and_unload()
            del t06_peft
            print("Instruction adapter merged successfully into base weights.", flush=True)
        elif config.sft_checkpoint_path and os.path.exists(
            os.path.join(config.sft_checkpoint_path, "adapter_config.json")
        ):
            # Trainable adapter present with no instruction in config/lineage —
            # allow legacy stacks, but make the risk explicit.
            print(
                "WARNING: loading SFT adapter without instruction_base_model / "
                "lineage instruction_adapter (raw-base stack).",
                flush=True,
            )

        if config.sft_checkpoint_path and os.path.exists(
            os.path.join(config.sft_checkpoint_path, "adapter_config.json")
        ):
            print(f"Loading SFT checkpoint strictly from {config.sft_checkpoint_path}...", flush=True)
            peft_config = config.build_peft_config()
            model = get_peft_model(base_model, peft_config)
            load_sft_adapter_strict(model, config.sft_checkpoint_path)
            model.train()
            peft_config = None
        elif required_instruction:
            print(
                f"Initializing fresh Rank-{config.lora_r} (alpha={config.lora_alpha}) "
                "LoRA adapter for GRPO training...",
                flush=True,
            )
            peft_config = config.build_peft_config()
            model = get_peft_model(base_model, peft_config)
            peft_config = None
        else:
            if config.sft_checkpoint_path:
                print(
                    f"Warning: Valid adapter_config.json not found in "
                    f"{config.sft_checkpoint_path}. Initializing new PEFT adapter on base model.",
                    flush=True,
                )
            peft_config = config.build_peft_config()
            model = get_peft_model(base_model, peft_config)
            peft_config = None

    grpo_config = config.build_grpo_config(include_model_init=False)

    use_extended_trainer = (
        config.zero_variance_strategy != "discard"
        or config.enable_crps
        or config.curriculum_schedule_type != "none"
    )

    if config.importance_sampling_level == "sequence_token" and use_extended_trainer:
        raise ValueError(
            "sequence_token is incompatible with direct scoring, CRPS, and the "
            "custom curriculum; set zero_variance_strategy='discard', "
            "enable_crps=false, and curriculum_schedule_type='none'"
        )

    if use_extended_trainer:
        from rlvr_pipeline.failure_mining_trainer import GRPOTrainerWithFailureMining as TrainerCls
    elif config.importance_sampling_level == "sequence_token":
        try:
            from trl.experimental.gspo_token import GRPOTrainer as TrainerCls
        except ImportError as e:
            raise ImportError(
                "GSPO-token requires trl.experimental.gspo_token. Ensure TRL is up to date."
            ) from e
    else:
        from trl import GRPOTrainer as TrainerCls

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        peft_config=peft_config,
        processing_class=processing_class,
    )

    if use_extended_trainer:
        trainer_kwargs["zero_variance_strategy"] = config.zero_variance_strategy
        trainer_kwargs["enable_crps"] = config.enable_crps
        trainer_kwargs["crps_max_age_steps"] = config.crps_max_age_steps
        trainer_kwargs["crps_max_traces"] = config.crps_max_traces

    print("Creating trainer (loading model)...", flush=True)
    trainer = TrainerCls(**trainer_kwargs)
    print(f"Trainer ready: {trainer.__class__.__name__}", flush=True)

    if fix_chat_template(getattr(trainer, "processing_class", None)):
        print("Chat template fixed: removed empty <think> injection (GRPO rollouts)", flush=True)

    _align_trainable_dtype_for_amp(trainer, fp16=config.fp16, bf16=config.bf16)

    if config.stop_strings:
        _attach_stop_string_criteria(trainer, config.stop_strings)

    if (
        use_extended_trainer
        and config.curriculum_schedule_type != "none"
        and hasattr(train_dataset, "column_names")
        and "difficulty_tag" in train_dataset.column_names
    ):
        trainer.attach_curriculum_sampler(config, train_dataset)
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            from collections import Counter as _Counter
            _log = _Path(r"c:\Users\Azooo\arabic-reasoning-rlvr-sota\debug-a273d4.log")
            _tags = list(train_dataset["difficulty_tag"]) if train_dataset is not None else []
            _known = {"trivial", "easy", "medium", "hard"}
            _counts = dict(_Counter(_tags))
            _orphan = {t: c for t, c in _counts.items() if t not in _known}
            with open(_log, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({
                    "sessionId": "a273d4", "hypothesisId": "B", "runId": "sanity",
                    "location": "trainer.py:curriculum_attach",
                    "message": "curriculum sampler attached",
                    "data": {
                        "attached": True,
                        "n": len(_tags),
                        "tag_counts": _counts,
                        "orphan_tags": _orphan,
                        "schedule": config.curriculum_schedule_type,
                    },
                    "timestamp": int(_time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
    elif config.curriculum_schedule_type != "none":
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            _log = _Path(r"c:\Users\Azooo\arabic-reasoning-rlvr-sota\debug-a273d4.log")
            _cols = list(getattr(train_dataset, "column_names", []) or [])
            with open(_log, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({
                    "sessionId": "a273d4", "hypothesisId": "B", "runId": "sanity",
                    "location": "trainer.py:curriculum_skip",
                    "message": "curriculum requested but NOT attached",
                    "data": {
                        "attached": False,
                        "use_extended_trainer": use_extended_trainer,
                        "has_difficulty_tag": "difficulty_tag" in _cols,
                        "columns": _cols[:20],
                        "schedule": config.curriculum_schedule_type,
                    },
                    "timestamp": int(_time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion

    if config.enable_stability_callback:
        from rlvr_pipeline.stability_callback import StabilityCallback

        stability_cb = StabilityCallback(
            consecutive=config.stability_consecutive,
            entropy_action=config.entropy_collapse_action,
            lr_reduction_factor=config.entropy_lr_reduction_factor,
            enable_adaptive_beta=config.enable_adaptive_beta,
            kl_near_zero_threshold=config.kl_near_zero_threshold,
            stuck_beta=config.stuck_beta,
            enable_adaptive_temperature=config.enable_adaptive_temperature,
            entropy_target_min=config.entropy_target_min,
            temp_bump=config.temp_bump,
            temp_max=config.temp_max,
            temp_bump_cooldown_steps=config.temp_bump_cooldown_steps,
            early_diag_steps=config.early_diag_steps,
        )
        stability_cb.trainer = trainer
        trainer.add_callback(stability_cb)

    # Attach automatic benchmark telemetry probe callback (opt-in; unsafe mid-GRPO on full GPUs)
    from rlvr_pipeline.benchmark_probe_callback import BenchmarkProbeCallback
    _probe_enabled = bool(getattr(config, "enable_benchmark_probe", False))
    probe_cb = BenchmarkProbeCallback(base_model=config.model_name, enable_probe=_probe_enabled)
    trainer.add_callback(probe_cb)
    if _probe_enabled:
        print("Benchmark probe ENABLED (will spawn vLLM on each save — ensure free VRAM).", flush=True)
    else:
        print("Benchmark probe disabled (enable_benchmark_probe=false).", flush=True)

    _maybe_attach_hub_checkpoint_callback(trainer, config)

    # Principal Engineer Fix: Integrate Arabic Terminal Reshaper directly into trainer._log_completions
    if hasattr(trainer, "_log_completions"):
        orig_log_comp = trainer._log_completions
        def custom_log_completions(*args, **kwargs):
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
                completions = kwargs.get("completions") or (args[1] if len(args) > 1 else [])
                if completions:
                    print("\n" + "="*80, flush=True)
                    print(" 📖 LIVE ARABIC GRPO COMPLETION ROLLOUT (RESHAPED & CONNECTED)", flush=True)
                    print("="*80, flush=True)
                    for i, comp in enumerate(completions[:2]):
                        text = comp[0]["content"] if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict) else str(comp)
                        reshaped = get_display(arabic_reshaper.reshape(text))
                        print(f"--- [Rollout #{i+1}] ---\n{reshaped}\n", flush=True)
                    print("="*80 + "\n", flush=True)
            except Exception:
                pass
            return orig_log_comp(*args, **kwargs)
        trainer._log_completions = custom_log_completions

    return trainer

