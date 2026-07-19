"""Top-tier DSPy signatures for Formal Arabic synthetic teacher CoT (SFT distillation).

Design (DSPy GEPA best practice):
- Short docstrings (GEPA rewrites instructions)
- Hard constraints in OutputField(desc=...) (stable under GEPA)
- Split think/answer fields (assemble XML in Module)
- Literal domain for typed I/O
"""
from __future__ import annotations

from typing import Literal

import dspy

Domain = Literal["gsm8k", "math", "math_comp", "logic"]
CritiqueSeverity = Literal["ok", "fixable", "reject"]


class PlanSolutionSteps(dspy.Signature):
    """Outline solution operations in Formal Arabic without stating the final answer."""

    domain: Domain = dspy.InputField(desc="gsm8k | math | math_comp | logic")
    problem: str = dspy.InputField(desc="Arabic word problem or logic puzzle text")
    plan: str = dspy.OutputField(
        desc=(
            "3-8 numbered Formal Arabic step labels naming operations only. "
            "FORBIDDEN: final numeric value, إذن…هو N, full logic JSON assignments, English."
        )
    )


class SolveTeacherCoT(dspy.Signature):
    """Produce Formal Arabic teacher reasoning and a canonical final answer for SFT."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    plan: str = dspy.InputField(desc="Operation plan to follow")
    think: str = dspy.OutputField(
        desc=(
            "Formal Arabic (فصحى) reasoning ONLY, ~60-180 words. "
            "Western digits 0-9 only (never ٠١٢٣٤٥٦٧٨٩). "
            "REQUIRED: show every computation explicitly as 'a × b = c' / 'a + b = c' / "
            "'a − b = c' / 'a ÷ b = c' (or 'يساوي') with the numeric result written; "
            "never leave 'a + b' or 'a × b' without '= c'; "
            "never say 'نحصل على الناتج' / 'بعد إيجاد ناتج' without writing that number. "
            "Numeric domains: at least one equation RHS must equal the canonical final value. "
            "Logic: name each assignment (X في المركز … / X هو …) explicitly. "
            "Complete every arithmetic step — never stop mid-sentence (e.g. 'ثم نضرب' / 'فيكون .'). "
            "ALLOWED: derived equation lines whose RHS equals the final value. "
            "FORBIDDEN: rhetorical closers إذن/لذلك/وبالتالي/الإجمالي … هو|هي <final> "
            "(without an equation); boilerplate 'نراجع الاتساق…'; English; ####; "
            "full logic ranking dump as JSON inside think."
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "Canonical final ONLY: ASCII digits for numeric domains, "
            "or compact flat JSON object string for logic. No units, no words."
        )
    )


class MaterializeExplicitArithmetic(dspy.Signature):
    """Rewrite vague CoT so every operation shows its numeric result (SFT-safe)."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField(desc="Exact required final answer")
    draft_think: str = dspy.InputField(
        desc="Draft think that may narrate ops without writing intermediate values"
    )
    draft_answer: str = dspy.InputField(desc="Draft canonical answer")
    think: str = dspy.OutputField(
        desc=(
            "Same solution method and Formal Arabic voice, but EVERY arithmetic/logic "
            "step MUST write the concrete value. Style: '8 × 4 = 32. ثم 6 × 20 = 120. "
            "ثم 32 + 120 = 152.' Keep Western digits. Keep ~60-180 words. "
            "Do NOT replace equations with 'نحصل على الناتج'. "
            "Do NOT add إذن/لذلك … هو <final> closers. "
            "Do NOT change the mathematical method or invent new facts. "
            "Logic: keep named role/rank assignments explicit in prose (not JSON)."
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth exactly")


class PolishFinishEquations(dspy.Signature):
    """Close dangling ops and ensure the final value appears as an equation RHS."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField(desc="Exact required final answer")
    draft_think: str = dspy.InputField(
        desc="Draft with possible bare ops, empty 'فيكون .', or missing final RHS"
    )
    draft_answer: str = dspy.InputField(desc="Draft canonical answer")
    think: str = dspy.OutputField(
        desc=(
            "Minimal edit of draft_think: (1) finish every bare 'a op b' as 'a op b = c'; "
            "(2) repair empty holes like 'فيكون .' with the real number; "
            "(3) ensure one explicit equation whose RHS equals ground_truth "
            "(e.g. '… = 28'); (4) keep Formal Arabic + Western digits; "
            "(5) do NOT add إذن/لذلك … هو <final> closers; "
            "(6) do NOT invent a new method — only close missing RHS values."
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth exactly")


class CritiqueTeacherDraft(dspy.Signature):
    """Critique a teacher draft against ground truth and ship quality rules."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField(desc="Exact required final answer")
    think: str = dspy.InputField(desc="Draft Formal Arabic reasoning")
    answer: str = dspy.InputField(desc="Draft canonical answer")
    issues: str = dspy.OutputField(
        desc=(
            "Bullet failures using tags: leak / wrong_answer / incomplete / boilerplate / "
            "eastern_digits / logic_dump / think_too_short / vague_narration / "
            "missing_intermediates / bare_op / missing_answer_eq / dangling_connector / ok. "
            "One issue per line. Use bare_op when 'a + b' lacks '= c'; "
            "missing_answer_eq when final digits never appear as an equation RHS."
        )
    )
    severity: CritiqueSeverity = dspy.OutputField(
        desc="ok if ship-ready; fixable if refine/strip/materialize/polish can help; reject if unusable"
    )


class RefineToGroundTruth(dspy.Signature):
    """Revise draft so the answer exactly matches ground truth; keep Formal Arabic think."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField(desc="Exact required final answer")
    draft_think: str = dspy.InputField(desc="Previous think text")
    draft_answer: str = dspy.InputField(desc="Previous answer")
    think: str = dspy.OutputField(
        desc=(
            "Revised Formal Arabic think. Western digits only. "
            "Write every intermediate as an explicit equation with its result. "
            "Do NOT end with إذن/لذلك … هو <final>. No الاتساق filler. No vague 'الناتج' without digits."
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth exactly")


class StripFinalLeak(dspy.Signature):
    """Delete only closing sentences that restate the final answer; keep intermediates."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    draft_think: str = dspy.InputField()
    draft_answer: str = dspy.InputField()
    think: str = dspy.OutputField(
        desc=(
            "Keep ALL explicit equation lines (including a op b = final_value). "
            "Remove ONLY rhetorical closers "
            "(إذن/لذلك/وبالتالي/الإجمالي/الجواب … هو|هي <final> without equation form). "
            "Do not invent new arithmetic. Do not insert filler. "
            "Do not delete intermediate numbers to 'hide' the answer."
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth")


class CompleteTruncatedSteps(dspy.Signature):
    """Finish dangling mid-operation think (e.g. ends at ثم نضرب) without restating final."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    draft_think: str = dspy.InputField(desc="Truncated Formal Arabic think")
    think: str = dspy.OutputField(
        desc=(
            "Completed think: finish ONLY the dangling calculation from the draft, "
            "writing the missing numeric results explicitly (a op b = c). "
            "Repair 'فيكون .' holes. Western digits. No إذن/لذلك closer. "
            "No new solution method. No الاتساق filler."
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth")


class CompactThink(dspy.Signature):
    """Shorten repetitive think while preserving Formal Arabic math and no final leak."""

    domain: Domain = dspy.InputField()
    problem: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    draft_think: str = dspy.InputField()
    think: str = dspy.OutputField(
        desc=(
            "~60-160 words; remove duplicated sentences; Western digits; "
            "KEEP every essential a op b = c line including the final RHS; "
            "do not convert equations into vague narration; do not drop bare-op closures"
        )
    )
    answer: str = dspy.OutputField(desc="Must equal ground_truth")


class FlashNumericAnswer(dspy.Signature):
    """Independent solve. Return the final numeric answer only as digits."""

    problem: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Digits only, no units, no words")


class FlashMathCompAnswer(dspy.Signature):
    """Independent solve for competition-style math. Return the final answer string."""

    problem: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer string (simplified)")


class FlashLogicAnswer(dspy.Signature):
    """Independent solve for logic puzzle. Return a flat JSON object of assignments."""

    problem: str = dspy.InputField()
    answer: str = dspy.OutputField(desc='Flat JSON object string, e.g. {"a":"x","b":"y"}')


# Back-compat aliases (older imports / partial GEPA loads)
PlanArabicSolution = PlanSolutionSteps
SolveFormalArabicCoT = SolveTeacherCoT
