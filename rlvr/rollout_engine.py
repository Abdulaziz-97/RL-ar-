"""
Module 2 — Rollout Engine (Generation).

Given a prompt, sample G completions from the current policy.
"""

from typing import Optional, Protocol, runtime_checkable, Union

DEFAULT_MAX_COMPLETION_LENGTH = 2048


@runtime_checkable
class GenerativeModel(Protocol):
    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        ...


@runtime_checkable
class GenerativeModelWithLogProbs(Protocol):
    """Optional protocol: generate returns (completion, per-token log-probs)."""

    def generate_with_logprobs(
        self, prompt: str, temperature: float, max_new_tokens: int
    ) -> tuple[str, list[float]]:
        ...


def generate_group(
    model,
    prompt: str,
    group_size: int = 8,
    temperature: float = 0.9,
    max_new_tokens: int = DEFAULT_MAX_COMPLETION_LENGTH,
    return_logprobs: bool = False,
) -> Union[list[str], tuple[list[str], Optional[list[list[float]]]]]:
    """Sample ``group_size`` completions.

    Truncation is left to ``model.generate(..., max_new_tokens=...)`` so the
    limit is enforced in token units, not characters.

    When ``return_logprobs=True`` and the model implements
    ``generate_with_logprobs``, also returns per-completion token log-probs.
    Otherwise the second element is ``None``.
    """
    if group_size <= 0:
        return ([], None) if return_logprobs else []

    if return_logprobs and hasattr(model, "generate_with_logprobs"):
        use_logprobs = True
    elif hasattr(model, "generate"):
        use_logprobs = False
    else:
        raise TypeError(
            "model must implement .generate(prompt, temperature, max_new_tokens) -> str "
            "or .generate_with_logprobs(...) -> (str, list[float])"
        )

    stripped = prompt.strip()
    is_empty = stripped == ""
    prompt_to_send = "" if is_empty else prompt

    completions: list[str] = []
    all_log_probs: list[list[float]] = []

    for _ in range(group_size):
        try:
            if use_logprobs:
                completion, log_probs = model.generate_with_logprobs(
                    prompt_to_send, temperature, max_new_tokens
                )
                all_log_probs.append(list(log_probs))
            else:
                completion = model.generate(prompt_to_send, temperature, max_new_tokens)
        except Exception:
            if is_empty:
                empty = [""] * group_size
                return (empty, None) if return_logprobs else empty
            raise
        # Do NOT slice by character count — max_new_tokens is a token limit
        # already enforced by the generation call.
        completions.append(completion)

    if return_logprobs:
        return completions, (all_log_probs if use_logprobs else None)
    return completions
