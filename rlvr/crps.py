"""
Module — CRPS (Curriculum Replay via Progressive Suffixes).

During the hard curriculum stage, when the model fails on a prompt, construct a
"progressive suffix" from the model's own past successful trajectory on a related
prompt. The suffix is prepended to the failed rollout's context as an in-context
correction hint for subsequent rollouts. The suffix starts small (just the final
conclusion) and grows with repeated failures, giving progressively stronger hints
only after the model has repeatedly failed.
"""

from dataclasses import dataclass


@dataclass
class SuccessTrace:
    prompt_id: str
    puzzle_type: str
    difficulty_tag: str
    token_sequence: str  # fully decoded completion text (preserves subword joins)
    step_recorded: int


class SuccessTraceStore:
    def __init__(self, max_traces: int = 256):
        self.traces: list[SuccessTrace] = []
        self.max_traces = max_traces

    def add(self, trace: SuccessTrace) -> None:
        """Add a trace. If store exceeds max_traces, evict oldest (FIFO)."""
        self.traces.append(trace)
        if len(self.traces) > self.max_traces:
            self.traces.pop(0)

    def __len__(self) -> int:
        return len(self.traces)

    def clear(self) -> None:
        self.traces.clear()


def find_related_success_trace(
    traces: list[SuccessTrace],
    puzzle_type: str,
    difficulty_tag: str,
    max_age_steps: int,
    current_step: int,
) -> SuccessTrace | None:
    """Find the most recent successful trace matching puzzle_type and difficulty_tag,
    within max_age_steps of the current step. Returns None if no match found."""
    candidates = [
        t
        for t in traces
        if t.puzzle_type == puzzle_type
        and t.difficulty_tag == difficulty_tag
        and (current_step - t.step_recorded) <= max_age_steps
    ]
    return candidates[-1] if candidates else None


def build_progressive_suffix(trace: SuccessTrace, suffix_fraction: float) -> str:
    """Take an increasing-length character suffix of the stored successful trace.

    Operates on the fully decoded string so BPE/WordPiece subwords (especially
    Arabic morphology) are not broken by force-joining token pieces with spaces.
    suffix_fraction in [0.0, 1.0].
    """
    text = trace.token_sequence
    if not text:
        return ""
    if isinstance(text, list):
        # Backward-compat for any leftover list-of-token traces.
        text = "".join(text)
    n_chars = max(1, int(len(text) * suffix_fraction))
    return text[-n_chars:]


def get_crps_suffix_fraction(failure_count: int) -> float:
    """Schedule: start small (0.1) and increase with repeated failures.
    failure_count=0 -> 0.0 (no hint yet, first attempt)
    failure_count=1 -> 0.1 (just the conclusion)
    failure_count=2 -> 0.3
    failure_count=3 -> 0.5
    failure_count>=4 -> 0.5 (cap at 0.5 to preserve genuine problem-solving)"""
    schedule = [0.0, 0.1, 0.3, 0.5, 0.5]
    return min(0.5, schedule[min(failure_count, 4)])


def should_activate_crps(
    curriculum_stage: str,
    has_success_trace: bool,
    failure_count: int,
) -> bool:
    """CRPS activates ONLY during the hard curriculum stage, ONLY when a prior success trace exists,
    and ONLY after at least 1 failure (failure_count >= 1)."""
    return curriculum_stage == "hard" and has_success_trace and failure_count >= 1
