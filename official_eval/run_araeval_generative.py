"""Generation-based Arabic evaluation runner for AraEval benchmarks.

This script evaluates Arabic language models using generation mode (not
log-likelihood) on the official AraEval MCQ tasks. The model generates
its answer with optional thinking/reasoning, and the answer is extracted
and graded against the gold label.

Usage:
    python run_araeval_generative.py \
        --model unsloth/Qwen3.5-4B \
        --output-dir /workspace/outputs/generative_eval_base

    python run_araeval_generative.py \
        --model unsloth/Qwen3.5-4B \
        --adapter-path aziz9788/qwen3.5-4b-arabic-grpo-v2 \
        --enable-thinking \
        --max-lora-rank 128 \
        --output-dir /workspace/outputs/generative_eval_grpo_v2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.araeval.generative_utils import (
    GENERATIVE_TASKS,
    build_generative_prompt,
    extract_answer,
    get_generation_profile,
    grade_answer,
    load_generative_checkpoint,
    new_generative_checkpoint,
    pending_generative_tasks,
    save_generative_checkpoint,
    summarize_generative_checkpoint,
)
from tasks.araeval.utils import (
    normalize_ien_mcq,
    normalize_ien_tf,
    normalize_aramath,
    normalize_etec,
    normalize_arapro,
    normalize_truthfulqa,
)

DEFAULT_MODEL = "unsloth/Qwen3.5-4B"

# Hugging Face dataset paths matching official YAML configs
DATASET_CONFIGS = {
    "araeval_ien_mcq": {
        "path": "humain-ai/IEN_MCQ",
        "revision": "a3ddadfbe6ed009001be9f1cecd4a61cff979337",
        "split": "test",
        "normalizer": normalize_ien_mcq,
    },
    "araeval_ien_tf": {
        "path": "humain-ai/IEN_TF",
        "revision": "1d2d4bcd3a08173bef4a29eb2530f505a99f0812",
        "split": "test",
        "normalizer": normalize_ien_tf,
    },
    "araeval_aramath": {
        "path": "humain-ai/AraMath",
        "revision": "b79e79b1b993d153613d1ce2357a1177f2cdcfa1",
        "split": "test",
        "normalizer": normalize_aramath,
    },
    "araeval_etec": {
        "path": "humain-ai/Etec",
        "revision": "61a79b9c2d8528e57d14adb145ff7a85261654e7",
        "split": "test",
        "normalizer": normalize_etec,
    },
    "araeval_arapro": {
        "path": "humain-ai/AraPro",
        "revision": "e606730ed50b663d7655ea2e02793238f5166422",
        "split": "test",
        "name": "AraPro",
        "normalizer": normalize_arapro,
    },
    "araeval_truthfulqa": {
        "path": "humain-ai/AraTruthfulQA",
        "revision": "162744fbf0590606415eb0924f2b5bd680486e3a",
        "split": "test",
        "normalizer": normalize_truthfulqa,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation-based AraEval evaluation with vLLM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "outputs" / "generative_eval",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-lora-rank", type=int, default=128)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning generation (recommended for GRPO models)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of specific tasks to evaluate (e.g. --tasks araeval_aramath araeval_ifeval)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for generation (e.g. --max-new-tokens 1024)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for debugging)",
    )
    return parser.parse_args(argv)


def load_dataset_for_task(task: str) -> list[dict[str, Any]]:
    """Load and normalize a dataset from Hugging Face for a given task."""
    from datasets import load_dataset

    config = DATASET_CONFIGS[task]
    kwargs = {
        "path": config["path"],
        "split": config["split"],
        "revision": config["revision"],
        "trust_remote_code": True,
    }
    if "name" in config:
        kwargs["name"] = config["name"]

    dataset = load_dataset(**kwargs)
    normalizer = config["normalizer"]

    # Normalize each row to get {query, choices, gold}
    normalized = []
    for row in dataset:
        try:
            norm = normalizer(row)
            normalized.append(norm)
        except Exception as e:
            print(f"  WARNING: Skipping row due to normalization error: {e}")
            continue

    return normalized


def _auto_merge_adapter_if_needed(args: argparse.Namespace) -> None:
    if not args.adapter_path or not os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        return
    merged_dir = os.path.join(os.path.dirname(args.adapter_path), f"merged_eval_{os.path.basename(args.adapter_path)}")
    if not os.path.exists(os.path.join(merged_dir, "model.safetensors")) and not os.path.exists(os.path.join(merged_dir, "model-00001-of-00002.safetensors")):
        print(f"Merging LoRA adapter ({args.adapter_path}) into base model ({args.model}) for 100% vLLM compatibility...", flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cpu")
        peft = PeftModel.from_pretrained(base, args.adapter_path)
        merged = peft.merge_and_unload()
        if hasattr(merged.config, "architectures") and merged.config.architectures == ["Qwen3_5ForCausalLM"]:
            merged.config.architectures = ["Qwen2ForCausalLM"]
            merged.config.model_type = "qwen2"
        merged.save_pretrained(merged_dir)
        tok_source = args.adapter_path if os.path.exists(os.path.join(args.adapter_path, "tokenizer_config.json")) else args.model
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        tokenizer.save_pretrained(merged_dir)

        # Patch saved config.json explicitly
        cfg_path = os.path.join(merged_dir, "config.json")
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("architectures") == ["Qwen3_5ForCausalLM"] or cfg.get("model_type") == "qwen3_5":
                cfg["architectures"] = ["Qwen2ForCausalLM"]
                cfg["model_type"] = "qwen2"
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
        print(f"Merged model saved & patched successfully to {merged_dir}", flush=True)

    args.model = merged_dir
    args.adapter_path = None


def build_vllm_engine(args: argparse.Namespace, task: str):
    """Build a vLLM LLM engine for generation."""
    from vllm import LLM, SamplingParams

    _auto_merge_adapter_if_needed(args)
    profile = get_generation_profile(task)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 1

    engine_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.max_length,
        "gpu_memory_utilization": profile["gpu_memory_utilization"],
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_num_seqs": profile.get("max_num_seqs", 64),
        "max_num_batched_tokens": 8192,
    }

    # Enable LoRA if adapter path is provided
    if args.adapter_path is not None:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = args.max_lora_rank

    llm = LLM(**engine_kwargs)

    max_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else profile["max_new_tokens"]
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=profile["temperature"],
        top_p=1.0,
    )

    return llm, sampling_params, profile


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    """Apply the model's chat template to a prompt."""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return formatted
    except TypeError:
        # Some tokenizers don't support enable_thinking kwarg
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted


def evaluate_task(
    llm,
    sampling_params,
    tokenizer,
    task: str,
    samples: list[dict[str, Any]],
    enable_thinking: bool,
    adapter_path: str | None,
) -> dict[str, Any]:
    """Evaluate a single task using generation mode."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"  Formatting {len(samples)} prompts...", flush=True)

    # Build generation prompts
    prompts = []
    gold_indices = []
    for sample in samples:
        query = sample["query"]
        gold = sample["gold"]
        prompts.append(
            apply_chat_template(tokenizer, query, enable_thinking)
        )
        gold_indices.append(gold)

    print(f"  Running generation on {len(prompts)} prompts...", flush=True)

    # Build LoRA request if adapter is provided
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest("grpo_v2_adapter", 1, adapter_path)

    # Generate all prompts with vLLM's native continuous batching engine
    if lora_request is not None:
        all_outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )
    else:
        all_outputs = llm.generate(prompts, sampling_params)

    # Extract and grade answers
    correct = 0
    extraction_failures = 0
    for output, gold_idx in zip(all_outputs, gold_indices):
        generated_text = output.outputs[0].text
        extracted = extract_answer(generated_text)
        if extracted is None:
            extraction_failures += 1
        if grade_answer(extracted, gold_idx):
            correct += 1

    total = len(samples)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 2),
        "extraction_failures": extraction_failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Main evaluation loop."""
    import torch

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"

    # Load or create checkpoint
    checkpoint = load_generative_checkpoint(checkpoint_path)
    if checkpoint is None or checkpoint.get("version") != 1:
        checkpoint = new_generative_checkpoint(
            model=args.model,
            adapter_path=args.adapter_path,
            enable_thinking=args.enable_thinking,
        )
        save_generative_checkpoint(checkpoint_path, checkpoint)

    all_tasks = args.tasks if args.tasks is not None else GENERATIVE_TASKS
    pending = [t for t in pending_generative_tasks(checkpoint) if t in all_tasks]

    if not pending:
        summary = summarize_generative_checkpoint(checkpoint)
        save_generative_checkpoint(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    print(
        f"Generative evaluation: {len(pending)} tasks pending for {args.model}",
        flush=True,
    )
    if args.tasks:
        print(f"  Selected tasks: {args.tasks}", flush=True)
    if args.max_new_tokens:
        print(f"  Max new tokens override: {args.max_new_tokens}", flush=True)
    if args.adapter_path:
        print(f"  LoRA adapter: {args.adapter_path}", flush=True)
    print(f"  Thinking: {'ON' if args.enable_thinking else 'OFF'}", flush=True)

    current_llm = None
    current_profile_key = None

    for task in pending:
        position = GENERATIVE_TASKS.index(task) + 1
        print(f"\n[{position}/{len(GENERATIVE_TASKS)}] {task}", flush=True)

        profile = get_generation_profile(task)
        profile_key = task if task in ("araeval_aramath", "araeval_arapro", "araeval_ifeval") else "gen_mcq"

        # Reload engine if profile changed
        if current_llm is None or current_profile_key != profile_key:
            if current_llm is not None:
                del current_llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"  Initializing vLLM engine with profile [{profile_key}]...", flush=True)
            current_llm, sampling_params, _ = build_vllm_engine(args, task)
            current_profile_key = profile_key

        # Load dataset
        print(f"  Loading dataset from HuggingFace...", flush=True)
        samples = load_dataset_for_task(task)
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"  Loaded {len(samples)} samples", flush=True)

        # Get tokenizer for chat template
        tokenizer = current_llm.get_tokenizer()

        # Run evaluation
        started = time.monotonic()
        result = evaluate_task(
            llm=current_llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            task=task,
            samples=samples,
            enable_thinking=args.enable_thinking,
            adapter_path=args.adapter_path,
        )
        result["seconds"] = round(time.monotonic() - started, 2)

        print(
            f"  Result: {result['correct']}/{result['total']} "
            f"({result['accuracy']}%) "
            f"[{result['extraction_failures']} extraction failures] "
            f"in {result['seconds']:.1f}s",
            flush=True,
        )

        # Save checkpoint
        checkpoint["completed"][task] = result
        save_generative_checkpoint(checkpoint_path, checkpoint)

    # Generate final summary
    summary = summarize_generative_checkpoint(checkpoint)
    save_generative_checkpoint(summary_path, summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
