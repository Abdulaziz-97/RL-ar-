"""Comparative evaluation script for AraMath across SFT base model and GRPO checkpoints."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "official_eval"))

from tasks.araeval.generative_utils import extract_answer, grade_answer, _LABELS
from tasks.araeval.utils import normalize_aramath
from datasets import load_dataset

def load_aramath_samples(limit: int | None = 50):
    dataset = load_dataset(
        "humain-ai/AraMath",
        revision="b79e79b1b993d153613d1ce2357a1177f2cdcfa1",
        split="test",
        trust_remote_code=True,
    )
    samples = []
    for row in dataset:
        norm = normalize_aramath(row)
        samples.append(norm)
    if limit is not None:
        samples = samples[:limit]
    return samples

def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {prompt}\nAssistant:"

def evaluate_checkpoint(model_name: str, adapter_path: str | None, samples: list[dict]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n=======================================================", flush=True)
    print(f"  Evaluating: {model_name} | Adapter: {adapter_path}", flush=True)
    print(f"=======================================================", flush=True)

    print("[1/3] Loading base model on GPU (SDPA Flash Attention)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"[2/3] Loading LoRA adapter ({adapter_path}) and zeroing embedding deltas...", flush=True)
        model = PeftModel.from_pretrained(base, adapter_path)
        for name, param in model.named_parameters():
            if "lora_" in name and ("embed_tokens" in name or "lm_head" in name):
                param.data.zero_()
        merged_model = model.merge_and_unload()
    else:
        print("[2/3] Using base Phase 1 SFT model without adapter...", flush=True)
        merged_model = base

    merged_model.eval()

    prompts = [apply_chat_template(tokenizer, s["query"]) for s in samples]
    gold_indices = [s["gold"] for s in samples]

    print(f"[3/3] Running high-speed PyTorch HF generation (batch_size=16) on {len(prompts)} samples...", flush=True)
    batch_size = 16
    all_texts = []
    start_time = time.monotonic()

    for b in range(0, len(prompts), batch_size):
        batch = prompts[b : b + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=1280,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_texts.append(gen_text)

    elapsed = round(time.monotonic() - start_time, 2)
    correct = 0
    failures = 0

    for text, gold_idx in zip(all_texts, gold_indices):
        extracted = extract_answer(text)
        if extracted is None:
            failures += 1
        if grade_answer(extracted, gold_idx):
            correct += 1

    total = len(samples)
    acc = round(100.0 * correct / total, 2) if total > 0 else 0.0

    print(f"\nResult for {adapter_path or model_name}:")
    print(f"  Accuracy: {correct}/{total} ({acc}%)")
    print(f"  Extraction Failures: {failures}")
    print(f"  Elapsed Time: {elapsed}s\n")

    # Clean up GPU RAM
    del merged_model
    del base
    torch.cuda.empty_cache()

    return {
        "model": adapter_path or model_name,
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "failures": failures,
        "seconds": elapsed,
    }

def main():
    parser = argparse.ArgumentParser(description="Fast AraMath Evaluation for GRPO checkpoint-100 and checkpoint-200")
    parser.add_argument("--base-model", type=str, default="aziz9788/T06__qwen35-mixed-v6-lr1e5")
    parser.add_argument("--checkpoints-dir", type=str, default="/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    print("\nLoading AraMath test set...", flush=True)
    samples = load_aramath_samples(limit=args.limit)
    print(f"Loaded {len(samples)} test samples for AraMath.", flush=True)

    results = []

    # Evaluate GRPO checkpoint-100 and checkpoint-200 strictly
    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        ckpt_dir = Path("/workspace/outputs/qwen_4b_2x5090_v4_run")

    target_ckpts = ["checkpoint-200"]
    for ckpt_name in target_ckpts:
        p = ckpt_dir / ckpt_name
        if p.exists() and (p / "adapter_config.json").exists():
            print(f"\n[EVAL] Evaluating GRPO {ckpt_name} (on base {args.base_model}) from {p}...", flush=True)
            res = evaluate_checkpoint(args.base_model, str(p), samples)
            results.append(res)
        else:
            print(f"\n[SKIP] {p} does not exist.")

    print("\n" + "=" * 65)
    print("        [ARAMATH EVALUATION: CHECKPOINT-200 ON AZIZ9788 BASE]        ")
    print("=" * 65)
    print(f"{'Checkpoint / Base':<45} | {'Accuracy':<10} | {'Time (s)':<8}")
    print("-" * 68)
    for r in results:
        name = Path(r["model"]).name
        print(f"{name:<45} | {r['accuracy']:>6.1f}%   | {r['seconds']:>8.1f}s")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
