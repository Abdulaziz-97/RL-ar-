from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PipelineConfig:
    group_size: int = 8
    temperature: float = 0.9
    max_completion_length: int = 2048

    w_correctness: float = 0.6
    w_format: float = 0.2
    w_language: float = 0.2
    w_answer_leak_penalty: float = 0.5
    w_structural_leak_penalty: float = 0.3

    clip_eps: float = 0.2
    kl_coef: float = 0.04
    eps: float = 1e-4

    clip_ratio: float = 0.02
    top_k_ratio: float = 0.02
    kl_cov_coef: float = 1.0

    policy_update_mode: Literal["grpo", "gspo"] = "grpo"
    entropy_guard_mode: Literal["clip_cov", "kl_cov"] = "clip_cov"
    curriculum_stage: Literal["easy", "hard"] = "easy"

    target_language: str = "ar"
    max_preamble_words: int = 5
    answer_leak_threshold: float = 0.75

    n_difficulty_attempts: int = 20


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
