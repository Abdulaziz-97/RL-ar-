"""
Curriculum-aware data sampling for E2H Reasoner Gaussian schedule.

Provides a Sampler that dynamically weights sample selection by difficulty
bucket, transitioning from easy to hard across training steps.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler
from transformers.trainer_callback import TrainerCallback

from rlvr.curriculum import (
    get_curriculum_weights,
    sample_curriculum_bucket,
)

BUCKET_NAMES = ("trivial", "easy", "medium", "hard")


class CurriculumSampler(Sampler[int]):
    """Weights sample selection by difficulty bucket using E2H Gaussian schedule.

    At each step, computes bucket probabilities via get_curriculum_weights(),
    then samples indices from the corresponding bucket.
    """

    def __init__(
        self,
        difficulty_tags: Sequence[str],
        total_steps: int,
        schedule_type: str = "gaussian",
        sigma_fraction: float = 0.2,
        num_samples: int | None = None,
        seed: int = 42,
        mini_repeat_count: int = 1,
        batch_size: int = 1,
        repeat_count: int = 1,
    ):
        self.difficulty_tags = list(difficulty_tags)
        self.total_steps = max(total_steps, 1)
        self.schedule_type = schedule_type
        self.sigma_fraction = sigma_fraction
        self.num_samples = num_samples or len(self.difficulty_tags)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self._rng = random.Random(seed)
        self._current_step = 0

        self._bucket_indices: dict[str, list[int]] = {b: [] for b in BUCKET_NAMES}
        for i, tag in enumerate(self.difficulty_tags):
            if tag in self._bucket_indices:
                self._bucket_indices[tag].append(i)
        self._all_indices = list(range(len(self.difficulty_tags)))

    def set_step(self, step: int) -> None:
        self._current_step = step

    def __iter__(self) -> Iterator[int]:
        num_batches = self.num_samples // self.batch_size
        for _ in range(num_batches):
            weights = get_curriculum_weights(
                self.schedule_type, self._current_step, self.total_steps, self.sigma_fraction
            )
            available = {
                bucket: weight
                for bucket, weight in weights.items()
                if self._bucket_indices.get(bucket)
            }
            total = sum(available.values())
            if total <= 0:
                normalized = None
            else:
                normalized = {bucket: weight / total for bucket, weight in available.items()}

            chunk = []
            for _ in range(self.batch_size):
                bucket = sample_curriculum_bucket(normalized, self._rng) if normalized else None
                pool = self._bucket_indices.get(bucket, []) if bucket is not None else self._all_indices
                chunk.append(self._rng.choice(pool))

            # Preserve TRL's RepeatSampler structure: each prompt must appear
            # num_generations times, and generated batches are reused together.
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        complete = (self.num_samples // self.batch_size) * self.batch_size
        return complete * self.mini_repeat_count * self.repeat_count

    def get_bucket_distribution(self) -> dict[str, float]:
        return get_curriculum_weights(
            self.schedule_type, self._current_step, self.total_steps, self.sigma_fraction
        )


class CurriculumCallback(TrainerCallback):
    """Updates the CurriculumSampler's step counter each optimizer step."""

    def __init__(self, sampler: CurriculumSampler):
        self.sampler = sampler

    def on_train_begin(self, args, state, control, **kwargs):
        self.sampler.set_step(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.sampler.set_step(state.global_step)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.sampler.set_step(state.global_step)
