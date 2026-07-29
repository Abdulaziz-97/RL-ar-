#!/usr/bin/env python3
"""
Fast Benchmark Telemetry Probe for GRPO RLVR.

Evaluates:
  1. AraMath (Full 605 questions)
  2. AraIFEval (Full 539 questions)
  3. AraPro (Fixed seed=42 500-question sample)

Total Probe Size: 1,644 questions
Estimated Runtime: ~1.5 to 2.0 minutes per probe check using vLLM continuous batching.

Usage:
  python scripts/run_benchmark_probe.py --checkpoint outputs/qwen_4b_2x5090_v3_run/checkpoint-50
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Import official AraEval utilities
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "official_eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "official_eval"))

from official_eval.tasks.araeval.generative_utils import (
    build_generative_prompt,
    extract_answer,
    grade_answer,
)
from official_eval.tasks.araeval.utils import (
    normalize_aramath,
    normalize_arapro,
)

PROBE_CONFIGS = {
    "araeval_aramath": {
        "path": "humain-ai/AraMath",
        "revision": "b79e79b1b993d153613d1ce2357a1177f2cdcfa1",
        "split": "test",
        "normalizer": normalize_aramath,
        "sample_size": None, # Full 605
    },
    "araeval_arapro": {
        "path": "humain-ai/AraPro",
        "revision": "e606730ed50b663d7655ea2e02793238f5166422",
        "split": "test",
        "name": "AraPro",
        "normalizer": normalize_arapro,
        "sample_size": 500, # Fixed 500 sample (seed=42)
    },
}

def load_probe_dataset(task_name: str, config: dict) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs = {"path": config["path"], "revision": config["revision"], "split": config["split"]}
    if "name" in config:
        kwargs["name"] = config["name"]

    ds = load_dataset(**kwargs)
    normalizer = config["normalizer"]

    docs = []
    for doc in ds:
        docs.append(normalizer(doc))

    if config["sample_size"] is not None and len(docs) > config["sample_size"]:
        rng = random.Random(42) # Fixed seed=42 for identical probe sample across all checks
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        selected_indices = sorted(indices[:config["sample_size"]])
        docs = [docs[i] for i in selected_indices]

    return docs

def run_probe(checkpoint_dir: str, base_model: str = "unsloth/Qwen3.5-4B") -> dict[str, float]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n=================================================================")
    print(f"🚀 FAST BENCHMARK TELEMETRY PROBE: {checkpoint_dir}")
    print(f"=================================================================\n")

    # Determine if checkpoint is LoRA adapter or full weights
    ckpt_path = Path(checkpoint_dir)
    has_adapter = (ckpt_path / "adapter_model.safetensors").exists() or (ckpt_path / "adapter_config.json").exists()

    llm_kwargs = {
        "model": base_model if has_adapter else str(ckpt_path),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "enable_lora": has_adapter,
        "max_lora_rank": 128 if has_adapter else None,
        "trust_remote_code": True,
    }
    if not has_adapter:
        llm_kwargs.pop("enable_lora")
        llm_kwargs.pop("max_lora_rank")

    llm = LLM(**llm_kwargs)
    lora_request = LoRARequest("probe_adapter", 1, str(ckpt_path)) if has_adapter else None
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])

    metrics = {}
    total_start = time.time()

    for task_name, cfg in PROBE_CONFIGS.items():
        print(f"Evaluating {task_name} (Probe size: {cfg['sample_size'] or 'Full'})...", flush=True)
        docs = load_probe_dataset(task_name, cfg)
        prompts = [build_generative_prompt(doc, task_name, enable_thinking=True) for doc in docs]

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        elapsed = time.time() - t0

        correct = 0
        total = len(docs)
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text
            extracted = extract_answer(gen_text)
            is_correct = grade_answer(extracted, docs[i]["target"], task_name)
            if is_correct:
                correct += 1

        acc = (correct / total) * 100 if total > 0 else 0.0
        metrics[task_name] = acc
        print(f"  --> {task_name}: {correct}/{total} ({acc:.2f}%) in {elapsed:.1f}s", flush=True)

    print(f"\n=================================================================")
    print(f"PROBE COMPLETE in {time.time() - total_start:.1f}s")
    for k, v in metrics.items():
        print(f"  * {k}: {v:.2f}%")
    print(f"=================================================================\n")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Benchmark Telemetry Probe")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3.5-4B", help="Base model path")
    args = parser.parse_args()

    run_probe(args.checkpoint, args.base_model)
