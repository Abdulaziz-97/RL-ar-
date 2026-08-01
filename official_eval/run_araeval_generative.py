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
    normalize_araifeval,
    process_ifeval_results,
)

_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MODEL = "unsloth/Qwen3.5-4B"


def normalize_ifeval_wrapper(row: dict[str, Any]) -> dict[str, Any]:
    norm = normalize_araifeval(row)
    return {
        "query": norm["prompt"],
        "choices": [],
        "gold": norm,
        "is_ifeval": True,
    }


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
    "araeval_ifeval": {
        "path": "humain-ai/AraIFEval",
        "revision": "1adcaee4cbbd253f9fe5dcf85b2e98c566a7b738",
        "split": "test",
        "normalizer": normalize_ifeval_wrapper,
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

    # Wipe stale unpatched directory if rope_scaling or rope_parameters or mrope_section exists in config
    cfg_path = os.path.join(merged_dir, "config.json")
    if os.path.exists(cfg_path):
        import json, shutil
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_txt = f.read()
            if "rope_scaling" in raw_txt or "rope_parameters" in raw_txt or "mrope_section" in raw_txt or "qwen3_5" in raw_txt or "Qwen3_5" in raw_txt:
                print(f"Wiping stale unpatched merge directory {merged_dir}...", flush=True)
                shutil.rmtree(merged_dir, ignore_errors=True)
        except Exception:
            pass

    if not os.path.exists(os.path.join(merged_dir, "model.safetensors")) and not os.path.exists(os.path.join(merged_dir, "model-00001-of-00002.safetensors")):
        print(f"Merging LoRA adapter ({args.adapter_path}) into base model ({args.model}) for 100% vLLM compatibility...", flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from safetensors.torch import save_file

        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cpu")
        peft = PeftModel.from_pretrained(base, args.adapter_path)

        # Zero out embed_tokens and lm_head LoRA parameters to prevent embedding distortion during merge
        for name, param in peft.named_parameters():
            if "lora_" in name and ("embed_tokens" in name or "lm_head" in name):
                param.data.zero_()

        merged = peft.merge_and_unload()

        def is_valid_qwen2_param(k: str) -> bool:
            if k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                return True
            if not k.startswith("model.layers."):
                return False
            parts = k.split(".")
            if len(parts) < 4:
                return False
            sub = parts[3]
            if sub in ("input_layernorm", "post_attention_layernorm"):
                return True
            if sub == "mlp" and len(parts) >= 5 and parts[4] in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
                return True
            if sub == "self_attn" and len(parts) >= 5 and parts[4] in ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                return True
            return False

        # Remap state dict keys in RAM before writing to disk once (prevents Errno 28 disk full)
        state_dict = merged.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("language_model.", "").replace("model.model.", "model.")
            if is_valid_qwen2_param(new_k):
                new_state_dict[new_k] = v.clone()

        os.makedirs(merged_dir, exist_ok=True)
        save_file(new_state_dict, os.path.join(merged_dir, "model.safetensors"))
        print(f"Saved cleanly remapped weights to {merged_dir}/model.safetensors", flush=True)

        merged.config.architectures = ["Qwen2ForCausalLM"]
        merged.config.model_type = "qwen2"
        num_layers = getattr(merged.config, "num_hidden_layers", 32)
        merged.config.max_window_layers = num_layers
        for attr in ["rope_scaling", "rope_parameters", "mrope_section"]:
            if hasattr(merged.config, attr):
                setattr(merged.config, attr, None)
        merged.config.save_pretrained(merged_dir)

        tok_source = args.adapter_path if os.path.exists(os.path.join(args.adapter_path, "tokenizer_config.json")) else args.model
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        tokenizer.save_pretrained(merged_dir)

    # Always ensure saved config.json and generation_config.json are completely purged of mrope/rope_scaling for vLLM
    def _recursive_purge_rope(d):
        if isinstance(d, dict):
            d.pop("rope_scaling", None)
            d.pop("rope_parameters", None)
            d.pop("mrope_section", None)
            d.pop("use_sliding_window", None)
            d.pop("sliding_window", None)
            for v in list(d.values()):
                _recursive_purge_rope(v)

    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        _recursive_purge_rope(cfg)
        cfg["architectures"] = ["Qwen2ForCausalLM"]
        cfg["model_type"] = "qwen2"
        num_layers = cfg.get("num_hidden_layers", 32)
        cfg["max_window_layers"] = num_layers

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Recursively purged & patched {cfg_path} to Qwen2ForCausalLM text config", flush=True)

    gen_cfg_path = os.path.join(merged_dir, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        import json
        try:
            with open(gen_cfg_path, "r", encoding="utf-8") as f:
                gen_cfg = json.load(f)
            _recursive_purge_rope(gen_cfg)
            with open(gen_cfg_path, "w", encoding="utf-8") as f:
                json.dump(gen_cfg, f, indent=2)
        except Exception:
            pass

    args.model = merged_dir
    args.adapter_path = None


def build_hf_engine(args: argparse.Namespace):
    """Build a PyTorch HuggingFace Transformers engine with merged weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[1/2] Loading base model ({args.model}) in bfloat16 on GPU...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.adapter_path and os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        print(f"[2/2] Loading GRPO LoRA adapter ({args.adapter_path}) and zeroing embedding deltas...", flush=True)
        model = PeftModel.from_pretrained(base, args.adapter_path)
        for name, param in model.named_parameters():
            if "lora_" in name and ("embed_tokens" in name or "lm_head" in name):
                param.data.zero_()
        merged_model = model.merge_and_unload()
    else:
        merged_model = base

    merged_model.eval()
    return merged_model, tokenizer


def evaluate_task(
    model,
    tokenizer,
    task: str,
    samples: list[dict[str, Any]],
    enable_thinking: bool,
) -> dict[str, Any]:
    """Evaluate a single task using PyTorch HuggingFace batched generation."""
    import torch

    profile = get_generation_profile(task)
    max_tokens = profile["max_new_tokens"]

    print(f"  Formatting {len(samples)} prompts...", flush=True)
    prompts = []
    gold_indices = []
    for sample in samples:
        query = sample["query"]
        gold = sample["gold"]
        prompts.append(apply_chat_template(tokenizer, query, enable_thinking))
        gold_indices.append(gold)

    print(f"  Running PyTorch HF batched generation on {len(prompts)} prompts...", flush=True)
    batch_size = 8
    all_generated_texts = []

    for b in range(0, len(prompts), batch_size):
        batch_prompts = prompts[b : b + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for out in output_ids:
            gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_generated_texts.append(gen_text)

    # Extract and grade answers
    correct = 0
    extraction_failures = 0
    for i, (generated_text, gold_idx, sample) in enumerate(zip(all_generated_texts, gold_indices, samples)):
        if i < 3:
            print(f"\n--- [DEBUG SAMPLE {i+1}] ---", flush=True)
            print(f"Generated text snippet: {repr(generated_text[:350])}", flush=True)
            if not sample.get("is_ifeval"):
                ext = extract_answer(generated_text)
                gold_label = _LABELS[gold_idx] if isinstance(gold_idx, int) and 0 <= gold_idx < len(_LABELS) else gold_idx
                print(f"Extracted: {ext} | Gold: {gold_label}", flush=True)
            print("---------------------------\n", flush=True)

        if sample.get("is_ifeval"):
            ifeval_res = process_ifeval_results(gold_idx, [generated_text])
            if ifeval_res.get("prompt_level_strict_acc"):
                correct += 1
        else:
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

    model, tokenizer = build_hf_engine(args)

    for task in pending:
        position = GENERATIVE_TASKS.index(task) + 1
        print(f"\n[{position}/{len(GENERATIVE_TASKS)}] {task}", flush=True)

        # Load dataset
        print(f"  Loading dataset from HuggingFace...", flush=True)
        samples = load_dataset_for_task(task)
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"  Loaded {len(samples)} samples", flush=True)

        # Run evaluation
        started = time.monotonic()
        result = evaluate_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            samples=samples,
            enable_thinking=args.enable_thinking,
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
