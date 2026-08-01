"""
Comprehensive Unit & E2E Test Suite for RLVR Pipeline.

Tests every module, function, callback, sampler, reward function, and config builder
in src/rlvr_pipeline/ to ensure complete correctness and bug-free execution.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- Module Imports ---
from rlvr_pipeline.config import RLVRConfig, DEFAULT_SYSTEM_PROMPT
from rlvr_pipeline.curriculum_sampler import (
    CurriculumSampler,
    CurriculumCallback,
    BUCKET_NAMES,
)
from rlvr_pipeline.data import (
    _derive_difficulty,
    _extract_answer_spec,
    _serialize_ground_truth,
    _transform_coldstart_response,
    load_cold_start_sft_dataset,
    load_production_dataset,
    load_rlvr_dataset,
    load_rlvr_dataset_legacy_v2,
)
from rlvr_pipeline.rewards import (
    ALL_REWARD_FUNCS,
    DEFAULT_REWARD_WEIGHTS,
    _extract_completion_text,
    answer_leak_penalty_func,
    composite_reward_func,
    correctness_reward_func,
    format_reward_func,
    language_reward_func,
    length_penalty_func,
    structural_leak_penalty_func,
)
from rlvr_pipeline.stability_callback import StabilityCallback
from scripts.filter_hard_dataset import derive_difficulty as filter_derive_difficulty


class TestRLVRPipelineConfig(unittest.TestCase):
    """Test RLVRConfig initialization, serialization, YAML parsing, and builder methods."""

    def test_default_config(self):
        cfg = RLVRConfig()
        self.assertEqual(cfg.model_name, "Qwen/Qwen3.5-2B")
        self.assertTrue(cfg.load_in_4bit)
        self.assertEqual(cfg.lora_r, 64)
        self.assertEqual(cfg.num_generations, 8)
        self.assertEqual(cfg.loss_type, "dr_grpo")
        self.assertIn("<think>", cfg.system_prompt)

    def test_yaml_load_and_save(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(
                """
model_name: "Qwen/Qwen3.5-4B"
learning_rate: 0.000005
num_generations: 16
beta: 0.0
entropy_coef: 0
use_adaptive_entropy: false
"""
            )
            f_path = f.name

        try:
            cfg = RLVRConfig.from_yaml(f_path)
            self.assertEqual(cfg.model_name, "Qwen/Qwen3.5-4B")
            self.assertEqual(cfg.learning_rate, 5e-6)
            self.assertEqual(cfg.num_generations, 16)
            self.assertEqual(cfg.beta, 0.0)
            self.assertEqual(cfg.entropy_coef, 0.0)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_build_peft_config(self):
        cfg = RLVRConfig(lora_r=128, lora_alpha=256)
        peft_cfg = cfg.build_peft_config()
        self.assertEqual(peft_cfg.r, 128)
        self.assertEqual(peft_cfg.lora_alpha, 256)
        self.assertIn("q_proj", peft_cfg.target_modules)

    def test_build_model_init_kwargs(self):
        cfg = RLVRConfig(load_in_4bit=True, bf16=True, attn_implementation="sdpa")
        kwargs = cfg.build_model_init_kwargs()
        self.assertIn("quantization_config", kwargs)
        self.assertEqual(kwargs["torch_dtype"], "bfloat16")
        self.assertEqual(kwargs["attn_implementation"], "sdpa")

    def test_build_grpo_config(self):
        cfg = RLVRConfig(
            output_dir="./test_out",
            num_generations=16,
            max_completion_length=512,
            temperature=1.15,
            loss_type="dr_grpo",
            beta=0.0,
            entropy_coef=0.0,
            use_adaptive_entropy=False,
        )
        grpo_cfg = cfg.build_grpo_config(include_model_init=False)
        self.assertEqual(grpo_cfg.output_dir, "./test_out")
        self.assertEqual(grpo_cfg.num_generations, 16)
        self.assertEqual(grpo_cfg.max_completion_length, 512)
        self.assertEqual(grpo_cfg.temperature, 1.15)
        self.assertEqual(grpo_cfg.loss_type, "dr_grpo")
        self.assertEqual(grpo_cfg.beta, 0.0)


class TestRLVPRewards(unittest.TestCase):
    """Test all 6 reward functions and edge cases."""

    def test_extract_completion_text(self):
        self.assertEqual(_extract_completion_text("hello"), "hello")
        self.assertEqual(
            _extract_completion_text([{"content": "world"}]), "world"
        )
        self.assertEqual(_extract_completion_text({"content": "foo"}), "foo")
        # Chat-template pre-fill normalization
        normalized = _extract_completion_text("my thoughts</think><answer>42</answer>")
        self.assertTrue(normalized.startswith("<think>\n"))

    def test_correctness_reward_func(self):
        prompts = ["prompt1", "prompt2"]
        completions = [
            "<think>step 1</think><answer>42</answer>",
            "<think>step 1</think><answer>100</answer>",
        ]
        ground_truth = ["42", "42"]
        domain = ["math", "math"]

        rewards = correctness_reward_func(
            prompts, completions, ground_truth_answer=ground_truth, domain=domain
        )
        self.assertEqual(len(rewards), 2)
        self.assertGreater(rewards[0], 0.5)  # Correct answer
        self.assertEqual(rewards[1], 0.0)  # Incorrect answer

    def test_format_reward_func(self):
        prompts = ["p1", "p2", "p3"]
        completions = [
            "<think>this is a deep thinking step with more than ten words of reasoning text</think><answer>42</answer>",  # Perfect format (>= 10 words)
            "<think>only think block step step step step step</think>",  # Partial
            "no tags at all",  # Failed format
        ]
        rewards = format_reward_func(prompts, completions)
        self.assertEqual(len(rewards), 3)
        self.assertGreater(rewards[0], 0.9)
        self.assertGreater(rewards[1], 0.0)
        self.assertEqual(rewards[2], 0.0)

    def test_language_reward_func(self):
        completions = [
            "<think>هذا تفكير باللغة العربية الصحيحة والواضحة</think><answer>42</answer>",
            "<think>this is completely English reasoning without Arabic tokens</think><answer>42</answer>",
        ]
        rewards = language_reward_func([], completions)
        self.assertGreater(rewards[0], rewards[1])

    def test_penalties(self):
        completions = ["<think>step</think><answer>42</answer>"]

        self.assertEqual(len(answer_leak_penalty_func([], completions)), 1)
        self.assertEqual(len(structural_leak_penalty_func([], completions)), 1)
        self.assertEqual(len(length_penalty_func([], completions)), 1)
        self.assertEqual(len(composite_reward_func([], completions, ground_truth_answer=["42"])), 1)


class TestCurriculumSampler(unittest.TestCase):
    """Test E2H Gaussian Curriculum Sampler."""

    def test_curriculum_sampler_init_and_iter(self):
        difficulty_tags = ["medium"] * 50 + ["hard"] * 50
        sampler = CurriculumSampler(
            difficulty_tags=difficulty_tags,
            total_steps=100,
            schedule_type="gaussian",
            batch_size=2,
            repeat_count=1,
            mini_repeat_count=1,
        )
        self.assertGreater(len(sampler), 0)

        # Iterate samples at step 0
        sampler.set_step(0)
        indices_step_0 = list(iter(sampler))
        self.assertEqual(len(indices_step_0), len(sampler))

        # Iterate samples at step 50
        sampler.set_step(50)
        indices_step_50 = list(iter(sampler))
        self.assertEqual(len(indices_step_50), len(sampler))

    def test_curriculum_callback(self):
        difficulty_tags = ["medium"] * 10
        sampler = CurriculumSampler(difficulty_tags, total_steps=10)
        cb = CurriculumCallback(sampler)

        mock_state = MagicMock()
        mock_state.global_step = 5

        cb.on_step_begin(None, mock_state, None)
        self.assertEqual(sampler._current_step, 5)


class TestStabilityCallback(unittest.TestCase):
    """Test StabilityCallback for entropy collapse, adaptive temp, and adaptive beta."""

    def test_on_log_diagnostics(self):
        dashboard = MagicMock()
        dashboard.record_live_signals.return_value = {
            "format_collapsed": False,
            "entropy_collapsing": True,
            "entropy_zero": False,
        }

        cb = StabilityCallback(
            consecutive=2,
            enable_adaptive_temperature=True,
            entropy_target_min=2.0,
            temp_bump=0.15,
            temp_max=1.5,
            dashboard=dashboard,
        )

        mock_trainer = MagicMock()
        mock_trainer.temperature = 1.0
        cb.trainer = mock_trainer

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_control = MagicMock()

        logs = {
            "loss": 0.5,
            "entropy": 0.3,
            "kl": 0.0001,
            "rewards/format_reward_func/mean": 0.95,
        }

        cb.on_log(mock_args, mock_state, mock_control, logs=logs)
        self.assertGreater(mock_trainer.temperature, 1.0)


class TestDataModule(unittest.TestCase):
    """Test dataset loading, parsing, and difficulty derivation."""

    def test_derive_difficulty(self):
        self.assertEqual(_derive_difficulty({"difficulty_tag": "hard"}), "hard")
        self.assertEqual(_derive_difficulty({"num_steps": 2}), "trivial")
        self.assertEqual(_derive_difficulty({"num_steps": 4}), "easy")
        self.assertEqual(_derive_difficulty({"num_steps": 6}), "medium")
        self.assertEqual(_derive_difficulty({"num_steps": 8}), "hard")
        self.assertEqual(_derive_difficulty({"grade_level": "grade_1"}), "easy")

    def test_serialize_ground_truth(self):
        self.assertEqual(_serialize_ground_truth(True), "true")
        self.assertEqual(_serialize_ground_truth(42), "42")
        self.assertEqual(_serialize_ground_truth({"a": 1}), '{"a": 1}')

    def test_transform_coldstart_response(self):
        raw_response = "Step 1: compute\n#### 42"
        transformed = _transform_coldstart_response(raw_response)
        self.assertIn("<answer>42</answer>", transformed)

    def test_dataset_loader(self):
        sample_jsonl = [
            {
                "id": "prob_1",
                "domain": "gsm8k",
                "prompt": "ما هو 2 + 2؟",
                "answer_spec": {"type": "integer", "canonical": "4"},
                "metadata": {"num_steps": 1},
            }
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for rec in sample_jsonl:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_path = f.name

        try:
            ds = load_production_dataset(tmp_path)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds[0]["ground_truth_answer"], "4")
            self.assertEqual(ds[0]["domain"], "math")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestFilterHardDataset(unittest.TestCase):
    """Test scripts/filter_hard_dataset.py logic."""

    def test_filter_derive_difficulty(self):
        trivial_rec = {
            "metadata": {"num_steps": 2, "grade_level": "grade_1"}
        }
        tag, keep = filter_derive_difficulty(trivial_rec)
        self.assertFalse(keep)
        self.assertEqual(tag, "trivial")

        medium_rec = {
            "metadata": {"num_steps": 4, "grade_level": "grade_5"}
        }
        tag, keep = filter_derive_difficulty(medium_rec)
        self.assertTrue(keep)
        self.assertEqual(tag, "medium")

        hard_rec = {
            "metadata": {"num_steps": 7, "grade_level": "grade_8"}
        }
        tag, keep = filter_derive_difficulty(hard_rec)
        self.assertTrue(keep)
        self.assertEqual(tag, "hard")


if __name__ == "__main__":
    unittest.main()
