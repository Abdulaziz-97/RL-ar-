"""
Module 6 — Failure Bank.

Bank all-zero-reward groups for replay during hard-stage curriculum.

Uses an instance-based store so experiments can be isolated. Module-level
helpers delegate to a default singleton for backward compatibility; call
``clear_bank()`` (or construct a fresh ``FailureBank``) at the start of every
experiment when running multiple jobs in one process.
"""

from collections import deque
from dataclasses import dataclass

MAX_BANK_SIZE = 10000


@dataclass
class BankedSample:
    prompt_id: str
    completions: list[str]
    rewards: list[float]
    stage: str


class FailureBank:
    """Instance-scoped failure bank (no cross-experiment leakage)."""

    def __init__(self, max_size: int = MAX_BANK_SIZE):
        self._bank: deque[BankedSample] = deque()
        self._current_stage: str = "easy"
        self.max_size = max_size

    def set_current_stage(self, stage: str) -> None:
        self._current_stage = stage

    def clear(self) -> None:
        self._bank.clear()
        self._current_stage = "easy"

    def __len__(self) -> int:
        return len(self._bank)

    def bank_failed_group(self, prompt_id: str, completions: list[str], rewards: list[float]) -> None:
        # Only bank groups where every reward is exactly 0.0.
        if not rewards:
            return
        if any(r != 0.0 for r in rewards):
            return

        sample = BankedSample(
            prompt_id=prompt_id,
            completions=list(completions),
            rewards=list(rewards),
            stage=self._current_stage,
        )
        self._bank.append(sample)
        while len(self._bank) > self.max_size:
            self._bank.popleft()

    def retrieve_banked_failures(self, stage: str) -> list:
        if stage == "hard":
            return list(self._bank)
        return []

    def evict_stale_banked_samples(self, active_prompt_ids: set[str]) -> int:
        kept = [s for s in self._bank if s.prompt_id in active_prompt_ids]
        removed = len(self._bank) - len(kept)
        self._bank.clear()
        self._bank.extend(kept)
        return removed


# Default singleton — prefer injecting FailureBank explicitly in new code.
_default_bank = FailureBank()


def set_current_stage(stage: str) -> None:
    _default_bank.set_current_stage(stage)


def clear_bank() -> None:
    """Reset bank contents and stage. Call at the start of every experiment."""
    _default_bank.clear()


def bank_size() -> int:
    return len(_default_bank)


def bank_failed_group(prompt_id: str, completions: list[str], rewards: list[float]) -> None:
    _default_bank.bank_failed_group(prompt_id, completions, rewards)


def retrieve_banked_failures(stage: str) -> list:
    return _default_bank.retrieve_banked_failures(stage)


def evict_stale_banked_samples(active_prompt_ids: set[str]) -> int:
    return _default_bank.evict_stale_banked_samples(active_prompt_ids)
