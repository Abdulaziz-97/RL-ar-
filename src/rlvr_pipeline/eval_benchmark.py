"""Evaluation engine for English and multilingual baseline benchmarks.

Runs zero-shot or few-shot inference on evaluation datasets (GSM8K, MATH, AraEval),
scores outputs using RLVR reward contracts (correctness, format, language), and
generates comprehensive diagnostic reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

from rlvr_pipeline.config import RLVRConfig
from rlvr_pipeline.data import load_rlvr_dataset
from rlvr_pipeline.rewards import reward_correctness, reward_format, reward_language
from rlvr_pipeline.trainer import fix_chat_template


def evaluate_benchmark_on_model(
    model_name_or_path: str,
    benchmark_data_path: str | Path,
    adapter_path: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 1.0,
    system_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Run model inference on a benchmark JSONL dataset and return evaluation metrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    data_path = Path(benchmark_data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {data_path}")

    dataset = load_rlvr_dataset(data_path, system_prompt=system_prompt)
    print(f"Loaded {len(dataset)} evaluation samples from {data_path.name}", flush=True)

    print(f"Loading tokenizer from {model_name_or_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    fix_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {model_name_or_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    prompts_raw = dataset["prompt"]
    answer_specs = dataset["answer_spec"]
    ground_truths = dataset["ground_truth"]
    sample_ids = dataset.get("sample_id", [f"item_{i}" for i in range(len(dataset))])
    domains = dataset.get("domain", ["general"] * len(dataset))

    formatted_prompts = []
    for prompt_msgs in prompts_raw:
        if isinstance(prompt_msgs, list):
            rendered = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = str(prompt_msgs)
        formatted_prompts.append(rendered)

    completions = []
    print("Generating completions...", flush=True)

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Evaluating"):
        batch_prompts = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        input_lengths = inputs.input_ids.shape[1]
        generated_tokens = output_ids[:, input_lengths:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        completions.extend(decoded)

    # Calculate rewards using canonical contracts
    print("Scoring outputs using RLVR reward functions...", flush=True)
    correctness_scores = reward_correctness(completions, answer_specs)
    format_scores = reward_format(completions)
    language_scores = reward_language(completions)

    correct_count = sum(1 for s in correctness_scores if s > 0.5)
    valid_format_count = sum(1 for s in format_scores if s > 0.5)
    valid_lang_count = sum(1 for s in language_scores if s > 0.5)
    total = len(dataset)

    mean_correctness = sum(correctness_scores) / total if total > 0 else 0.0
    mean_format = sum(format_scores) / total if total > 0 else 0.0
    mean_language = sum(language_scores) / total if total > 0 else 0.0

    completion_lens = [len(c.split()) for c in completions]
    avg_words = sum(completion_lens) / total if total > 0 else 0.0

    detailed_results = []
    for sid, p, c, gt, spec, dom, corr, fmt, lang in zip(
        sample_ids,
        formatted_prompts,
        completions,
        ground_truths,
        answer_specs,
        domains,
        correctness_scores,
        format_scores,
        language_scores,
    ):
        detailed_results.append({
            "sample_id": sid,
            "domain": dom,
            "ground_truth": gt,
            "completion": c,
            "correctness": corr,
            "format": fmt,
            "language": lang,
        })

    metrics = {
        "benchmark_file": data_path.name,
        "total_samples": total,
        "correctness_accuracy": mean_correctness,
        "format_accuracy": mean_format,
        "language_accuracy": mean_language,
        "exact_matches": correct_count,
        "avg_words_per_completion": avg_words,
        "details": detailed_results,
    }

    print("\n" + "=" * 50, flush=True)
    print(f" BENCHMARK EVALUATION RESULTS: {data_path.name}", flush=True)
    print("=" * 50, flush=True)
    print(f" Total Samples Evaluated : {total}", flush=True)
    print(f" Correctness Accuracy   : {mean_correctness * 100:.2f}% ({correct_count}/{total})", flush=True)
    print(f" Format Tag Accuracy    : {mean_format * 100:.2f}% ({valid_format_count}/{total})", flush=True)
    print(f" Language Compliance    : {mean_language * 100:.2f}% ({valid_lang_count}/{total})", flush=True)
    print(f" Avg Words per Completion: {avg_words:.1f}", flush=True)
    print("=" * 50 + "\n", flush=True)

    return metrics
