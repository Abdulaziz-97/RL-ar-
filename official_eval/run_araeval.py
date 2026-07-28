from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register Qwen3_5ForConditionalGeneration architecture in vLLM ModelRegistry
try:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    ModelRegistry.register_model("Qwen3_5ForConditionalGeneration", Qwen2ForCausalLM)
except Exception:
    pass

# Monkey-patch PretrainedConfig for Qwen3_5 pad_token_id compatibility
try:
    from transformers.configuration_utils import PretrainedConfig
    _orig_config_getattribute = PretrainedConfig.__getattribute__
    def _patched_config_getattribute(self, key):
        if key == "pad_token_id":
            try:
                val = _orig_config_getattribute(self, key)
                if val is not None:
                    return val
            except AttributeError:
                pass
            return getattr(self, "eos_token_id", 151643)
        return _orig_config_getattribute(self, key)
    PretrainedConfig.__getattribute__ = _patched_config_getattribute
except Exception:
    pass

from tasks.araeval.utils import (
    PUBLIC_TEST_COUNTS,
    RANDOM_BASELINES,
    araeval_overall,
    normalize_against_random,
)


DEFAULT_MODEL = "unsloth/Qwen3.5-4B"
DEFAULT_LOADER = "vllm"
VLLM_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}
TASKS = [
    "araeval_ien_mcq",
    "araeval_ien_tf",
    "araeval_aramath",
    "araeval_etec",
    "araeval_arapro",
    "araeval_truthfulqa",
    "araeval_ifeval",
]
FULL_PUBLIC_DOCUMENTS = sum(PUBLIC_TEST_COUNTS.values())

_SHORT_NAMES = {
    "araeval_ien_mcq": "ien_mcq",
    "araeval_ien_tf": "ien_tf",
    "araeval_aramath": "aramath",
    "araeval_etec": "etec",
    "araeval_arapro": "arapro",
    "araeval_truthfulqa": "truthfulqa",
}

_COUNT_KEYS = {
    **_SHORT_NAMES,
    "araeval_ifeval": "araifeval",
}

_IFEVAL_METRICS = {
    "prompt_strict": "prompt_level_strict_acc,none",
    "instruction_strict": "inst_level_strict_acc,none",
    "prompt_loose": "prompt_level_loose_acc,none",
    "instruction_loose": "inst_level_loose_acc,none",
}


def proportional_sample_counts(total: int) -> dict[str, int]:
    available = sum(PUBLIC_TEST_COUNTS.values())
    if not 1 <= total <= available:
        raise ValueError(f"sample size must be between 1 and {available}")

    allocation: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    for position, task in enumerate(TASKS):
        count = PUBLIC_TEST_COUNTS[_COUNT_KEYS[task]]
        allocation[task] = total * count // available
        remainders.append((total * count % available, -position, task))

    remaining = total - sum(allocation.values())
    for _, _, task in sorted(remainders, reverse=True)[:remaining]:
        allocation[task] += 1
    return allocation


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None and "," not in name:
        value = metrics.get(f"{name},none")
    if not isinstance(value, (int, float)):
        raise KeyError(f"Missing numeric metric {name!r}")
    return float(value)


def summarize_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    completed = checkpoint.get("completed", {})
    raw_percent: dict[str, float] = {}

    for task, short_name in _SHORT_NAMES.items():
        if task not in completed:
            continue
        raw_percent[short_name] = 100.0 * _metric(
            completed[task]["metrics"], "acc_norm"
        )

    ifeval = completed.get("araeval_ifeval")
    if ifeval:
        for variant, metric_name in _IFEVAL_METRICS.items():
            raw_percent[f"ifeval_{variant}"] = 100.0 * _metric(
                ifeval["metrics"], metric_name
            )

    normalized: dict[str, float] = {}
    for name in _SHORT_NAMES.values():
        if name in raw_percent:
            normalized[name] = normalize_against_random(
                raw_percent[name], RANDOM_BASELINES[name]
            )
    for variant in _IFEVAL_METRICS:
        key = f"ifeval_{variant}"
        if key in raw_percent:
            normalized[key] = raw_percent[key]

    overall: dict[str, float] = {}
    core_names = set(_SHORT_NAMES.values())
    if core_names.issubset(raw_percent):
        for variant in _IFEVAL_METRICS:
            ifeval_key = f"ifeval_{variant}"
            if ifeval_key not in raw_percent:
                continue
            scores = {name: raw_percent[name] for name in core_names}
            scores["araifeval"] = raw_percent[ifeval_key]
            overall[variant] = araeval_overall(scores)

    return {
        "benchmark_mode": checkpoint.get("benchmark_mode"),
        "full_public_documents": checkpoint.get("full_public_documents"),
        "selected_tasks": checkpoint.get("selected_tasks"),
        "model": checkpoint.get("model"),
        "adapter_path": checkpoint.get("adapter_path"),
        "limit": checkpoint.get("limit"),
        "sample_size": checkpoint.get("sample_size"),
        "seed": checkpoint.get("seed"),
        "sample_counts": checkpoint.get("sample_counts"),
        "completed_tasks": sorted(completed),
        "total_seconds": sum(
            float(task.get("seconds", task.get("elapsed_seconds", 0.0)))
            for task in completed.values()
        ),
        "raw_percent": raw_percent,
        "normalized_percent": normalized,
        "overall_normalized": overall,
        "paper_primary": overall.get("prompt_strict"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full checkpointed AraEval benchmark with vLLM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "outputs" / "araeval_full_qwen35_4b",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Run only this task; repeat the option to select multiple tasks.",
    )
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument(
        "--limit",
        type=int,
        help="Diagnostic per-task limit; omit for the reported full benchmark.",
    )
    subset.add_argument(
        "--sample-size",
        type=int,
        help="Diagnostic proportional sample; omit for the reported full benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-lora-rank", type=int, default=128)
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable thinking/reasoning generation (default: False for loglik tasks)")
    parser.add_argument("--disable-thinking", action="store_false", dest="enable_thinking", help="Disable thinking/reasoning generation")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Number of GPUs for tensor parallelism (default: auto-detect CUDA count)")
    return parser.parse_args(argv)


TASK_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    # Short MCQ tasks (Tasks 1, 2, 3, 4, 6): Fast batching (batch_size: 16)
    "fast_mcq": {
        "batch_size": 16,
        "max_batch_size": 16,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.75,
    },
    # AraPro (Task 5 Alone): Long medical/science passages -> Micro-batched (batch_size: 4) for 100% OOM safety
    "araeval_arapro": {
        "batch_size": 4,
        "max_batch_size": 4,
        "max_num_batched_tokens": 1024,
        "gpu_memory_utilization": 0.50,
    },
    # AraIFEval (Task 7): High-throughput generation batching (batch_size: 32) with 4096 max batched tokens
    "araeval_ifeval": {
        "batch_size": 32,
        "max_batch_size": 32,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.85,
    },
}


def build_vllm_kwargs(args: argparse.Namespace, task: str = None) -> dict[str, Any]:
    import torch
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 1
    enable_thinking = getattr(args, "enable_thinking", False)
    profile = TASK_HYPERPARAMETERS.get(task, TASK_HYPERPARAMETERS.get("fast_mcq"))
    
    kwargs = {
        "pretrained": args.model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "batch_size": profile["batch_size"],
        "max_batch_size": profile["max_batch_size"],
        "max_num_batched_tokens": profile["max_num_batched_tokens"],
        "max_length": args.max_length,
        "enable_thinking": enable_thinking,
        "language_model_only": True,
        "gpu_memory_utilization": profile["gpu_memory_utilization"],
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "lora_local_path": (
            str(args.adapter_path)
            if args.adapter_path is not None
            else None
        ),
        "max_lora_rank": args.max_lora_rank,
    }
    if enable_thinking:
        kwargs["think_end_token"] = "</think>"
    return kwargs


def benchmark_mode(args: argparse.Namespace) -> str:
    if args.sample_size is not None:
        return "sample"
    if args.limit is not None:
        return "per_task_limit"
    return "full"


def new_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    sample_counts = (
        proportional_sample_counts(args.sample_size)
        if args.sample_size is not None
        else None
    )
    selected_tasks = args.task or TASKS
    return {
        "version": 3,
        "benchmark_mode": benchmark_mode(args),
        "full_public_documents": FULL_PUBLIC_DOCUMENTS,
        "selected_tasks": list(selected_tasks),
        "model": args.model,
        "adapter_path": (
            str(args.adapter_path)
            if args.adapter_path is not None
            else None
        ),
        "limit": args.limit,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sample_counts": sample_counts,
        "settings": {
            "loader": DEFAULT_LOADER,
            "dtype": "bfloat16",
            "device": "cuda",
            "batch_size": args.batch_size,
            "max_batch_size": args.max_batch_size,
            "max_length": args.max_length,
            "max_lora_rank": args.max_lora_rank,
            "apply_chat_template": True,
            "enable_thinking": False,
            "num_fewshot": 0,
            "bootstrap_iters": 0,
            "enable_prefix_caching": False,
        },
        "completed": {},
    }


def _load_checkpoint(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file():
        checkpoint = new_checkpoint(args)
        _atomic_write_json(path, checkpoint)
        return checkpoint

    try:
        with open(path, "r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
    except Exception:
        checkpoint = new_checkpoint(args)
        _atomic_write_json(path, checkpoint)
        return checkpoint

    if checkpoint.get("version") != 3:
        checkpoint = new_checkpoint(args)
        _atomic_write_json(path, checkpoint)
        return checkpoint

    checkpoint["benchmark_mode"] = benchmark_mode(args)
    checkpoint["selected_tasks"] = list(args.task or TASKS)
    _atomic_write_json(path, checkpoint)
    return checkpoint


def pending_tasks(
    checkpoint: dict[str, Any],
    selected_tasks: list[str],
) -> list[str]:
    completed = checkpoint.get("completed", {})
    return [task for task in selected_tasks if task not in completed]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import gc
    import torch

    for name, value in VLLM_ENV.items():
        os.environ.setdefault(name, value)

    try:
        import lm_eval
        from lm_eval.models.vllm_causallms import VLLM
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError(
            "lm_eval or vllm is not installed. Please install vllm and lm_eval."
        ) from exc

    task_dir = ROOT / "tasks" / "araeval"
    if args.adapter_path is not None:
        adapter_str = str(args.adapter_path)
        adapter_path = Path(adapter_str)
        if adapter_path.exists() and adapter_path.is_dir():
            if not (adapter_path / "adapter_config.json").is_file():
                raise ValueError(f"Invalid local adapter directory: {adapter_path}")
            if not any(adapter_path.glob("adapter_model.*")):
                raise ValueError(f"Adapter weights are missing: {adapter_path}")
    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    checkpoint = _load_checkpoint(checkpoint_path, args)
    selected_tasks = args.task or TASKS
    if args.sample_size is not None and args.task:
        raise ValueError("--sample-size runs the complete seven-task suite")
    if checkpoint["sample_counts"] is not None:
        os.environ["ARAEVAL_SAMPLE_COUNTS"] = json.dumps(
            checkpoint["sample_counts"], sort_keys=True
        )
        os.environ["ARAEVAL_SAMPLE_SEED"] = str(args.seed)
    else:
        os.environ.pop("ARAEVAL_SAMPLE_COUNTS", None)
        os.environ.pop("ARAEVAL_SAMPLE_SEED", None)
    pending = pending_tasks(checkpoint, selected_tasks)

    if not pending:
        summary = summarize_checkpoint(checkpoint)
        _atomic_write_json(summary_path, summary)
        return summary

    task_manager = TaskManager(include_path=str(task_dir))
    current_model = None
    current_profile_key = None

    for task in pending:
        position = selected_tasks.index(task) + 1
        print(f"[{position}/{len(selected_tasks)}] {task}", flush=True)

        target_profile_key = task if task in TASK_HYPERPARAMETERS else "fast_mcq"
        if current_model is None or current_profile_key != target_profile_key:
            if current_model is not None:
                del current_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"Initializing vLLM engine with profile [{target_profile_key}] for {task}...", flush=True)
            current_model = VLLM(**build_vllm_kwargs(args, task=task))
            current_profile_key = target_profile_key

        started = time.monotonic()
        try:
            result = lm_eval.simple_evaluate(
                model=current_model,
                tasks=[task],
                num_fewshot=0,
                limit=args.limit,
                bootstrap_iters=0,
                log_samples=False,
                apply_chat_template=True,
                fewshot_as_multiturn=False,
                task_manager=task_manager,
            )
            if result is None:
                raise RuntimeError(f"LM Harness returned no result for {task}")
            metrics = dict(result["results"][task])
            checkpoint["completed"][task] = {
                "metrics": metrics,
                "n_samples": result.get("n-samples", {}).get(task),
                "seconds": time.monotonic() - started,
            }
            checkpoint.pop("last_error", None)
            _atomic_write_json(checkpoint_path, checkpoint)
            _atomic_write_json(summary_path, summarize_checkpoint(checkpoint))
        except Exception as exc:
            checkpoint["last_error"] = {
                "task": task,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _atomic_write_json(checkpoint_path, checkpoint)
            raise

    summary = summarize_checkpoint(checkpoint)
    _atomic_write_json(summary_path, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed tasks remain checkpointed.", file=sys.stderr)
        raise SystemExit(130)
