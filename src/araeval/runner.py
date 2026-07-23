"""
Main AraEval evaluation engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from araeval.config import AraEvalConfig
from araeval.evaluators import EvalResult, evaluate_ifeval, evaluate_mcq
from araeval.loader import AraEvalSample, load_araeval_task
from rlvr_pipeline.config import DEFAULT_SYSTEM_PROMPT
from rlvr_pipeline.trainer import fix_chat_template, load_sft_adapter_strict

logger = logging.getLogger(__name__)


class AraEvalRunner:
    """Evaluation runner for AraEval datasets."""

    def __init__(self, config: AraEvalConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self) -> None:
        """Load tokenizer and base model + LoRA adapter if specified."""
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

        torch_dtype = torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not quant_config else None,
            device_map=device_map,
            trust_remote_code=True,
        )

        if self.config.adapter_path:
            print(f"Loading LoRA adapter strictly from {self.config.adapter_path}...", flush=True)
            lora_cfg = LoraConfig.from_pretrained(self.config.adapter_path)
            lora_cfg.inference_mode = True
            peft_model = get_peft_model(self.model, lora_cfg)
            load_sft_adapter_strict(peft_model, self.config.adapter_path)
            self.model = peft_model

        self.model.eval()

    def generate_completion(self, prompt: str) -> str:
        """Generate response for a single prompt at temperature=0.0."""
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text

    def evaluate_task(self, task_name: str) -> dict[str, Any]:
        """Evaluate a single AraEval task."""
        samples = load_araeval_task(task_name, limit=self.config.limit)
        print(f"Evaluating task {task_name} ({len(samples)} samples)...", flush=True)

        results: list[EvalResult] = []
        correct_count = 0
        ifeval_inst_ratios = []

        for sample in samples:
            completion = self.generate_completion(sample.prompt)

            if task_name == "ara_ifeval":
                strict_pass, ratio, inst_meta = evaluate_ifeval(completion, sample.instructions)
                is_correct = strict_pass
                pred_str = "PASS" if strict_pass else "FAIL"
                ifeval_inst_ratios.append(ratio)
            else:
                is_correct, pred_str = evaluate_mcq(completion, sample.gold_answer, sample.options)

            if is_correct:
                correct_count += 1

            results.append(
                EvalResult(
                    sample_id=sample.id,
                    task_name=task_name,
                    is_correct=is_correct,
                    gold_answer=sample.gold_answer,
                    predicted_answer=pred_str,
                    raw_completion=completion,
                    score=1.0 if is_correct else 0.0,
                    metadata=sample.metadata,
                )
            )

        accuracy = (correct_count / len(samples)) if samples else 0.0
        task_report = {
            "task_name": task_name,
            "total_samples": len(samples),
            "correct_samples": correct_count,
            "accuracy": round(accuracy, 4),
            "results": [
                {
                    "sample_id": r.sample_id,
                    "is_correct": r.is_correct,
                    "gold": r.gold_answer,
                    "pred": r.predicted_answer,
                    "completion": r.raw_completion,
                }
                for r in results
            ],
        }

        if task_name == "ara_ifeval" and ifeval_inst_ratios:
            task_report["instruction_level_accuracy"] = round(
                sum(ifeval_inst_ratios) / len(ifeval_inst_ratios), 4
            )

        return task_report

    def run(self) -> dict[str, Any]:
        """Run AraEval across all configured tasks and return aggregate summary."""
        if self.model is None:
            self.load_model_and_tokenizer()

        task_list = self.config.get_task_list()
        summary = {"tasks": {}, "macro_accuracy": 0.0}
        accuracies = []

        for task_name in task_list:
            report = self.evaluate_task(task_name)
            summary["tasks"][task_name] = report
            accuracies.append(report["accuracy"])

        summary["macro_accuracy"] = (
            round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_out = out_dir / "araeval_results.json"
        json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nAraEval Complete! Macro Accuracy: {summary['macro_accuracy']:.4f}")
        print(f"Report saved to {json_out}")
        return summary


def evaluate_model(config: AraEvalConfig) -> dict[str, Any]:
    """Convenience entry point to run AraEval evaluation."""
    runner = AraEvalRunner(config)
    return runner.run()
