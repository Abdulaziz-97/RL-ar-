"""Student pass@N calibration (N=8) with band assignment.

Runs the *exact student protocol* when a generate_fn is provided. For
credential-free / GPU-free CI, a stub generate_fn may be injected.
Private-eval labels must remain isolated — this module never writes labels
into generation prompts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

GenerateFn = Callable[[str, int], str]  # (prompt, seed) -> completion text


@dataclass
class PassAtNResult:
    problem_id: str
    n: int
    passes: int
    pass_fraction: float
    band: str
    seeds: list[int] = field(default_factory=list)
    checkpoint_hash: str = ""
    tokenizer_id: str = ""
    system_prompt_hash: str = ""
    raw: list[bool] = field(default_factory=list)


def assign_band(pass_fraction: float) -> str:
    """easy / medium / hard / deferred / mastered bands from pass@8 fraction."""
    if pass_fraction >= 0.999:
        return "mastered"
    if pass_fraction >= 0.75:
        return "easy"
    if pass_fraction >= 0.375:
        return "medium"
    if pass_fraction >= 0.125:
        return "hard"
    return "deferred"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def calibrate_pass_at_n(
    problems: list[dict[str, Any]],
    *,
    generate_fn: GenerateFn,
    verify_fn: Callable[[dict[str, Any], str], bool],
    n: int = 8,
    base_seed: int = 0,
    checkpoint_hash: str = "",
    tokenizer_id: str = "",
    system_prompt: str = "",
) -> list[PassAtNResult]:
    """Record raw pass fraction for each problem under fixed seeds/protocol."""
    results: list[PassAtNResult] = []
    sys_hash = _hash_text(system_prompt) if system_prompt else ""
    for i, prob in enumerate(problems):
        # Never send private labels / answer_spec to the generator prompt builder here;
        # generate_fn receives only the user prompt string.
        prompt = str(prob.get("prompt", ""))
        seeds = [base_seed + i * n + k for k in range(n)]
        flags: list[bool] = []
        for seed in seeds:
            completion = generate_fn(prompt, seed)
            flags.append(bool(verify_fn(prob, completion)))
        passes = sum(flags)
        frac = passes / n if n else 0.0
        results.append(
            PassAtNResult(
                problem_id=str(prob.get("problem_id") or prob.get("id") or i),
                n=n,
                passes=passes,
                pass_fraction=frac,
                band=assign_band(frac),
                seeds=seeds,
                checkpoint_hash=checkpoint_hash,
                tokenizer_id=tokenizer_id,
                system_prompt_hash=sys_hash,
                raw=flags,
            )
        )
    return results


def write_calibration(results: list[PassAtNResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stub_generate_fn_factory(answer_by_prompt: dict[str, str] | None = None) -> GenerateFn:
    """Deterministic stub student for tests (no GPU/API)."""
    answers = answer_by_prompt or {}

    def _gen(prompt: str, seed: int) -> str:
        # Pseudo-student: correct on even seeds when answer known via side channel
        # is NOT used — instead encode a trivial heuristic for stub problems.
        # For tests, callers should pass verify_fn that knows the answer_spec.
        ans = answers.get(prompt)
        if ans is None:
            # Extract last integer in prompt if present.
            import re

            nums = re.findall(r"-?\d+", prompt)
            ans = str(sum(int(x) for x in nums[:2])) if len(nums) >= 2 else "0"
        # Fail on odd seeds to create nontrivial pass fractions.
        if seed % 2 == 1:
            return f"<think>\nخطأ متعمد\n</think>\n<answer>{int(ans) + 1 if str(ans).isdigit() else 'x'}</answer>"
        return f"<think>\nحل صحيح\nخطوة ثانية\nإذن الناتج = {ans}\n</think>\n<answer>{ans}</answer>"

    return _gen
