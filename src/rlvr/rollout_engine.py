"""
Module 2 — Rollout Engine (Generation).

Given a prompt, sample G completions from the current policy.
"""

from typing import Protocol, runtime_checkable

DEFAULT_MAX_COMPLETION_LENGTH = 2048


@runtime_checkable
class GenerativeModel(Protocol):
    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        ...


def generate_group(
    model,
    prompt: str,
    group_size: int = 8,
    temperature: float = 0.9,
    max_new_tokens: int = DEFAULT_MAX_COMPLETION_LENGTH,
) -> list[str]:
    if not isinstance(model, GenerativeModel):
        raise TypeError(
            "model must implement the GenerativeModel protocol "
            "(provide a .generate(prompt, temperature, max_new_tokens) -> str method)"
        )

    if group_size <= 0:
        return []

    stripped = prompt.strip()
    is_empty = stripped == ""
    prompt_to_send = "" if is_empty else prompt

    completions: list[str] = []
    for _ in range(group_size):
        try:
            completion = model.generate(prompt_to_send, temperature, max_new_tokens)
        except Exception:
            if is_empty:
                return [""] * group_size
            raise
        completions.append(completion[:max_new_tokens])

    return completions
