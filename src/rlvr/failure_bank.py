"""
Module 6 — Failure Bank.

Bank all-zero-reward groups for replay during hard-stage curriculum.
"""

from collections import deque
from dataclasses import dataclass

MAX_BANK_SIZE = 10000

_bank: "deque[BankedSample]" = deque()
_current_stage: str = "easy"


@dataclass
class BankedSample:
    prompt_id: str
    completions: list[str]
    rewards: list[float]
    stage: str


def set_current_stage(stage: str) -> None:
    global _current_stage
    _current_stage = stage


def clear_bank() -> None:
    _bank.clear()


def bank_size() -> int:
    return len(_bank)


def bank_failed_group(prompt_id: str, completions: list[str], rewards: list[float]) -> None:
    # Only bank groups where every reward is exactly 0.0.
    if not rewards:
        return
    if any(r != 0.0 for r in rewards):
        return

    sample = BankedSample(
        prompt_id=prompt_id,
        completions=list(completions),
        rewards=list(rewards),
        stage=_current_stage,
    )
    _bank.append(sample)
    # FIFO eviction once over capacity.
    while len(_bank) > MAX_BANK_SIZE:
        _bank.popleft()


def retrieve_banked_failures(stage: str) -> list:
    if stage == "hard":
        return list(_bank)
    return []


def evict_stale_banked_samples(active_prompt_ids: set[str]) -> int:
    kept = [s for s in _bank if s.prompt_id in active_prompt_ids]
    removed = len(_bank) - len(kept)
    _bank.clear()
    _bank.extend(kept)
    return removed
