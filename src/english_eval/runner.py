"""
Evaluation engine for running English baseline benchmarks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from english_eval.config import EnglishEvalConfig
from english_eval.evaluators import (
    EnglishEvalResult,
    evaluate_gsm8k,
    evaluate_ifeval_item,
    evaluate_mcq,
)
from english_eval.loader import EnglishEvalSample, load_english_task
from rlvr_pipeline.trainer import fix_chat_template, load_sft_adapter_strict

logger = logging.getLogger(__name__)


ENGLISH_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and precise AI assistant. "
    "Respond to the user's instructions cleanly and directly."
)


class EnglishEvalRunner:
    """Evaluation runner for English benchmarks."""

    def __init__(self, config: EnglishEvalConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self) -> None:
        """Load base model and optional LoRA adapter."""
        ckpt = self.config.adapter_path or self.config.model_name_or_path
        print(f"Loading tokenizer from {ckpt}...", flush=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path, trust_remote_code=True
            )
        fix_chat_template(self.tokenizer)

        print(f"Loading base model {self.config.model_name_or_path}...", flush=True)
        device_map = "auto" if torch.cuda.is_available() else None

        quant_config = None
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        torch_dtype = (
            torch.bfloat16
            if self.config.bf16
            else (torch.float16 if self.config.fp16 else torch.float32)
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not quant_config else None,
            device_map=device_map,
            trust_remote_code=True,
        )

        if self.config.adapter_path:
            print(f"Loading LoRA adapter from {self.config.adapter_path}...", flush=True)
            lora_cfg = LoraConfig.from_pretrained(self.config.adapter_path)
            lora_cfg.inference_mode = True
            peft_model = get_peft_model(self.model, lora_cfg)
            load_sft_adapter_strict(peft_model, self.config.adapter_path)
            self.model = peft_model

        self.model.eval()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of English prompts."""
        formatted_list = []
        for p in prompts:
            messages = [
                {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = f"User: {p}\nAssistant:"
            formatted_list.append(formatted)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(formatted_list, return_tensors="pt", padding=True).to(
            self.model.device
        )
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.config.temperature > 0:
                gen_kwargs["temperature"] = self.config.temperature
                gen_kwargs["top_p"] = self.config.top_p
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.model.generate(**inputs, **gen_kwargs)

        completions = []
        for i, out in enumerate(outputs):
            gen_tokens = out[input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            completions.append(text)
        return completions

    def evaluate_task(self, task_name: str) -> dict[str, Any]:
        """Run evaluation for a single English task."""
        samples = load_english_task(
            task_name,
            limit=self.config.limit,
            mmlu_subjects=self.config.mmlu_subjects,
        )
        if not samples:
            print(f"Warning: No samples found for task {task_name}", flush=True)
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0, "results": []}

        print(f"Evaluating {len(samples)} samples on {task_name}...", flush=True)

        results: list[EnglishEvalResult] = []
        batch_size = self.config.batch_size

        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            prompts = [s.prompt for s in batch_samples]
            completions = self.generate_batch(prompts)

            for sample, completion in zip(batch_samples, completions):
                if task_name == "ifeval":
                    is_correct, score, details = evaluate_ifeval_item(
                        completion, sample.instruction_ids, sample.kwargs_list
                    )
                    res = EnglishEvalResult(
                        sample_id=sample.sample_id,
                        task_name=task_name,
                        is_correct=is_correct,
                        gold_answer=sample.gold_answer,
                        predicted_answer=f"strict_pass={is_correct}",
                        raw_completion=completion,
                        score=score,
                        metadata={"rule_details": details, **sample.metadata},
                    )
                elif task_name in ("mmlu", "hellaswag"):
                    is_correct, pred_letter = evaluate_mcq(
                        completion, sample.gold_answer, sample.options
                    )
                    res = EnglishEvalResult(
                        sample_id=sample.sample_id,
                        task_name=task_name,
                        is_correct=is_correct,
                        gold_answer=sample.gold_answer,
                        predicted_answer=pred_letter,
                        raw_completion=completion,
                        score=1.0 if is_correct else 0.0,
                        metadata=sample.metadata,
                    )
                elif task_name == "gsm8k":
                    is_correct, pred_num, gold_num = evaluate_gsm8k(
                        completion, sample.gold_answer
                    )
                    res = EnglishEvalResult(
                        sample_id=sample.sample_id,
                        task_name=task_name,
                        is_correct=is_correct,
                        gold_answer=gold_num,
                        predicted_answer=pred_num,
                        raw_completion=completion,
                        score=1.0 if is_correct else 0.0,
                        metadata=sample.metadata,
                    )
                else:
                    res = EnglishEvalResult(
                        sample_id=sample.sample_id,
                        task_name=task_name,
                        is_correct=False,
                        gold_answer=sample.gold_answer,
                        predicted_answer="",
                        raw_completion=completion,
                        score=0.0,
                        metadata={},
                    )

                results.append(res)

        correct_count = sum(1 for r in results if r.is_correct)
        avg_score = (
            sum(r.score for r in results) / len(results) if results else 0.0
        )
        accuracy = correct_count / len(results) if results else 0.0

        return {
            "accuracy": accuracy,
            "avg_score": avg_score,
            "total_samples": len(results),
            "correct_samples": correct_count,
            "results": [r.__dict__ for r in results],
        }

    def run(self) -> dict[str, Any]:
        """Execute full English baseline evaluation suite."""
        if self.model is None:
            self.load_model_and_tokenizer()

        tasks = self.config.get_task_list()
        print(f"Starting English baseline evaluation on tasks: {tasks}", flush=True)

        summary: dict[str, Any] = {"tasks": {}}
        accuracies = []

        for t in tasks:
            t_res = self.evaluate_task(t)
            summary["tasks"][t] = t_res
            accuracies.append(t_res["accuracy"])

        summary["macro_accuracy"] = (
            sum(accuracies) / len(accuracies) if accuracies else 0.0
        )

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_file = output_path / "english_eval_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Results written to {summary_file}", flush=True)
        return summary
