#!/usr/bin/env python3
"""Final pre-run debug gate for V4. Writes NDJSON evidence to workspace debug-a273d4.log."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
WS_LOG = REPO.parent / "debug-a273d4.log"
ALT_LOG = REPO / "debug-a273d4.log"
SESSION = "a273d4"
RUN_ID = "post-fix"


def _log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str | None = None) -> None:
    payload = {
        "sessionId": SESSION,
        "runId": run_id or RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for path in (WS_LOG, ALT_LOG):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    failures: list[str] = []

    # ── H-A: SFT inherits GRPO max_steps=250 ─────────────────────────────
    from rlvr_pipeline.config import RLVRConfig
    from rlvr_pipeline.trainer import build_sft_trainer

    cfg = RLVRConfig.from_yaml(REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml")
    _log(
        "A",
        "prerun:config",
        "loaded V4 yaml",
        {
            "max_steps": cfg.max_steps,
            "sft_max_steps": cfg.sft_max_steps,
            "sft_num_train_epochs": cfg.sft_num_train_epochs,
            "beta": cfg.beta,
            "resume_policy": cfg.resume_policy,
            "zero_variance_strategy": cfg.zero_variance_strategy,
            "curriculum_schedule_type": cfg.curriculum_schedule_type,
            "enable_benchmark_probe": cfg.enable_benchmark_probe,
            "lora_targets": list(cfg.lora_target_modules),
            "reward_weights": list(cfg.reward_weights),
        },
    )
    if cfg.max_steps != 250:
        failures.append(f"H-A unexpected grpo max_steps={cfg.max_steps}")
    if cfg.sft_max_steps not in (None, 0) and cfg.sft_max_steps == 250:
        failures.append("H-A sft_max_steps equals GRPO 250")
    if cfg.sft_num_train_epochs is None or cfg.sft_num_train_epochs < 1:
        failures.append("H-A sft epochs missing")

    sft_kwargs = {}
    with patch("rlvr_pipeline.trainer.load_cold_start_sft_dataset") as mock_ds, patch(
        "trl.SFTTrainer"
    ) as mock_trainer_cls, patch("trl.SFTConfig") as mock_sft_cfg, patch(
        "transformers.AutoModelForCausalLM"
    ), patch("peft.get_peft_model", side_effect=lambda m, c: m), patch("peft.PeftModel"):
        mock_ds.return_value = MagicMock()
        mock_sft_cfg.return_value = MagicMock(max_steps=None)
        mock_trainer_cls.return_value = MagicMock(model=MagicMock(), processing_class=None)
        cfg2 = RLVRConfig.from_yaml(REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml")
        cfg2.instruction_base_model = None
        cfg2.coldstart_data_path = str(REPO / "data" / "arabic_reasoning_coldstart_v4.jsonl")
        try:
            build_sft_trainer(cfg2)
        except Exception as e:
            _log("A", "prerun:sft_build", "build_sft_trainer exception (may be ok)", {"error": str(e)})
        if mock_sft_cfg.called:
            sft_kwargs = dict(mock_sft_cfg.call_args.kwargs)
    _log(
        "A",
        "prerun:sft_config_kwargs",
        "SFTConfig construction kwargs",
        {
            "kwargs_max_steps": sft_kwargs.get("max_steps"),
            "num_train_epochs": sft_kwargs.get("num_train_epochs"),
            "called": bool(sft_kwargs),
        },
    )
    if sft_kwargs.get("max_steps") == 250:
        failures.append("H-A CONFIRMED: SFTConfig got max_steps=250")

    # ── H-B: fresh run still auto-resumes ────────────────────────────────
    from rlvr_pipeline import cli
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        out.mkdir()
        (out / "checkpoint-100").mkdir()
        (out / "checkpoint-100" / "adapter_config.json").write_text("{}", encoding="utf-8")
        cfg_path = Path(td) / "cfg.yaml"
        yaml.safe_dump(
            {
                "model_name": "tiny",
                "train_data_path": str(Path(td) / "train.jsonl"),
                "output_dir": str(out),
                "max_steps": 2,
                "num_generations": 2,
                "per_device_train_batch_size": 2,
                "load_in_4bit": False,
                "bf16": False,
                "fp16": False,
                "use_wandb": False,
                "enable_crps": False,
                "curriculum_schedule_type": "none",
                "zero_variance_strategy": "discard",
                "enable_stability_callback": False,
                "enable_benchmark_probe": False,
                "resume_policy": "fresh",
            },
            cfg_path.open("w", encoding="utf-8"),
        )
        (Path(td) / "train.jsonl").write_text(
            '{"prompt":"q","answer_spec":{"type":"integer","canonical":1},'
            '"metadata":{"ground_truth_answer":"1"},"domain":"math"}\n',
            encoding="utf-8",
        )
        trainer = MagicMock()
        trainer.train_dataset = [0]
        trainer.get_train_dataloader = MagicMock(return_value=[0, 0])
        trainer.args = MagicMock(
            max_steps=2,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=2,
        )
        with patch("rlvr_pipeline.cli.build_trainer", return_value=trainer):
            rc = cli.main(["train", "--config", str(cfg_path)])
        kwargs = trainer.train.call_args.kwargs if trainer.train.called else {}
        resumed = kwargs.get("resume_from_checkpoint")
        _log(
            "B",
            "prerun:cli_fresh",
            "fresh CLI train with existing checkpoint",
            {"rc": rc, "resume_from_checkpoint": resumed, "train_called": trainer.train.called},
        )
        if resumed:
            failures.append(f"H-B CONFIRMED: auto-resumed from {resumed}")

    # ── H-C: GRPOConfig batch math / generation_batch_size invalid ───────
    grpo = cfg.build_grpo_config(include_model_init=False)
    gen_batch = getattr(grpo, "generation_batch_size", None)
    steps_per_gen = getattr(grpo, "steps_per_generation", None)
    expected_steps = cfg.gradient_accumulation_steps
    _log(
        "C",
        "prerun:grpo_config",
        "GRPOConfig batch fields",
        {
            "per_device_train_batch_size": grpo.per_device_train_batch_size,
            "num_generations": grpo.num_generations,
            "generation_batch_size": gen_batch,
            "steps_per_generation": steps_per_gen,
            "expected_steps_per_generation": expected_steps,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_steps": getattr(grpo, "max_steps", None),
            "beta": getattr(grpo, "beta", None),
            "save_total_limit": getattr(grpo, "save_total_limit", None),
        },
    )
    if steps_per_gen != expected_steps:
        failures.append(
            f"H-C steps_per_generation={steps_per_gen} != grad_accum={expected_steps}"
        )
    if getattr(grpo, "max_steps", None) == 210:
        failures.append("H-C mystery max_steps=210 injected")
    if gen_batch is not None and gen_batch % cfg.num_generations != 0:
        failures.append("H-C generation_batch_size not divisible by num_generations")

    # Try constructing trainer kwargs path without model load: inspect source contract
    trainer_src = (REPO / "src" / "rlvr_pipeline" / "trainer.py").read_text(encoding="utf-8")
    uses_strict = "load_sft_adapter_strict(" in trainer_src
    bare_trainable = "PeftModel.from_pretrained(base_model, config.sft_checkpoint_path, is_trainable=True)" in trainer_src
    _log(
        "C2",
        "prerun:trainer_source",
        "strict SFT load wiring",
        {"uses_load_sft_adapter_strict": uses_strict, "bare_trainable_from_pretrained": bare_trainable},
    )
    if not uses_strict or bare_trainable:
        failures.append("H-C2 strict adapter load not wired / bare from_pretrained still present")

    # ── H-D: reward diagnose / data contracts ────────────────────────────
    from rlvr_pipeline.rewards import diagnose_reward_weights

    try:
        report = diagnose_reward_weights(cfg.reward_weights)
        _log("D", "prerun:rewards", "diagnose_reward_weights ok", {"report": report})
    except Exception as e:
        failures.append(f"H-D reward diagnose failed: {e}")
        _log("D", "prerun:rewards", "diagnose_reward_weights FAILED", {"error": str(e)})

    sft_path = REPO / "data" / "arabic_reasoning_coldstart_v4.jsonl"
    rlvr_path = REPO / "data" / "arabic_reasoning_rlvr_v4.jsonl"
    sft_n = sum(1 for _ in open(sft_path, encoding="utf-8")) if sft_path.exists() else -1
    rlvr_n = sum(1 for _ in open(rlvr_path, encoding="utf-8")) if rlvr_path.exists() else -1
    _log(
        "D",
        "prerun:data_counts",
        "dataset row counts",
        {"sft_exists": sft_path.exists(), "sft_n": sft_n, "rlvr_exists": rlvr_path.exists(), "rlvr_n": rlvr_n},
    )
    if sft_n != 4000:
        failures.append(f"H-D SFT rows={sft_n} expected 4000")
    if rlvr_n <= 0:
        failures.append(f"H-D RLVR rows={rlvr_n}")

    # Sample load without fail_closed first; then with env
    from rlvr_pipeline.data import load_rlvr_dataset
    import os

    try:
        ds = load_rlvr_dataset(rlvr_path, system_prompt=cfg.system_prompt, production=True)
        _log("D", "prerun:data_load", "production load", {"n": len(ds), "cols": list(ds.column_names)})
    except Exception as e:
        failures.append(f"H-D production data load failed: {e}")
        _log("D", "prerun:data_load", "production load FAILED", {"error": str(e)})

    # ── H-E: orchestration shell/orchestrator regressions ────────────────
    vast = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")
    orch = (REPO / "data_1" / "scripts" / "run_v4_master_orchestrator.py").read_text(encoding="utf-8")
    eval_sh = (REPO / "run_eval.sh").read_text(encoding="utf-8")
    e_data = {
        "has_output_flag": "--output" in vast,
        "has_GRPO_OUT": "GRPO_OUT" in vast,
        "has_pkill_python": "pkill -9 -f python" in vast,
        "orch_max_steps_100": "--max-steps 100" in orch,
        "eval_has_T06": "T06" in eval_sh,
        "eval_raw_qwen_only": "Qwen/Qwen3.5-4B" in eval_sh and "T06" not in eval_sh,
        "forbidden_lora": bool({"embed_tokens", "lm_head"} & set(cfg.lora_target_modules)),
        "probe_off": cfg.enable_benchmark_probe is False,
    }
    _log("E", "prerun:orchestration", "shell/orchestrator contracts", e_data)
    if e_data["has_pkill_python"]:
        failures.append("H-E pkill -9 -f python still present")
    if not e_data["has_output_flag"] or not e_data["has_GRPO_OUT"]:
        failures.append("H-E vastai missing explicit GRPO --output")
    if e_data["orch_max_steps_100"]:
        failures.append("H-E orchestrator still caps SFT at 100")
    if e_data["eval_raw_qwen_only"] or not e_data["eval_has_T06"]:
        failures.append("H-E run_eval.sh not using T06 lineage")
    if e_data["forbidden_lora"]:
        failures.append("H-E embed/lm_head still in targets")
    if not e_data["probe_off"]:
        failures.append("H-E benchmark probe still enabled")

    # ── H-F: preflight module / integrity import ─────────────────────────
    try:
        from rlvr_pipeline import preflight, checkpoint_integrity, lineage

        _log(
            "F",
            "prerun:modules",
            "core modules importable",
            {
                "preflight": bool(preflight),
                "checkpoint_integrity": bool(checkpoint_integrity),
                "lineage": bool(lineage),
                "vllm_imported": "vllm" in sys.modules,
            },
        )
        if "vllm" in sys.modules:
            failures.append("H-F vllm imported during training package use")
    except Exception as e:
        failures.append(f"H-F module import failed: {e}")
        _log("F", "prerun:modules", "import FAILED", {"error": str(e)})

    ok = len(failures) == 0
    _log(
        "SUMMARY",
        "prerun:summary",
        "pre-run gate result",
        {"ok": ok, "failures": failures},
    )
    print("PRE-RUN DEBUG GATE:", "PASS" if ok else "FAIL")
    for f in failures:
        print("  -", f)
    print(f"Logs: {WS_LOG} (and {ALT_LOG})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
