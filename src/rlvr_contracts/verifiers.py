"""Executable numeric and flat-logic verifiers with a versioned registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, Optional

from rlvr_contracts.answer_spec import (
    AnswerSpec,
    AnswerSpecError,
    canonicalize_answer,
    parse_answer_spec,
    reject_symbolic,
)
from rlvr_contracts.response import extract_answer

VERIFIER_REGISTRY_VERSION = "rlvr-contracts-verifiers-v1"


@dataclass(frozen=True)
class VerifierResult:
    ok: bool
    score: float
    reason: str
    predicted_canonical: Optional[str] = None
    expected_canonical: Optional[str] = None
    verifier_id: str = "unknown"
    verifier_version: str = VERIFIER_REGISTRY_VERSION


def _flat_logic_equal(a: Any, b: Any) -> bool:
    numeric_a = isinstance(a, (int, float)) and not isinstance(a, bool)
    numeric_b = isinstance(b, (int, float)) and not isinstance(b, bool)
    if type(a) is not type(b) and not (numeric_a and numeric_b):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_flat_logic_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_flat_logic_equal(x, y) for x, y in zip(a, b))
    if numeric_a and numeric_b:
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def verify_integer(predicted: str, spec: AnswerSpec) -> VerifierResult:
    try:
        pred = canonicalize_answer("integer", predicted)
        exp = spec.canonical
        ok = pred == exp
        return VerifierResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="match" if ok else "integer_mismatch",
            predicted_canonical=pred,
            expected_canonical=exp,
            verifier_id="numeric.integer",
        )
    except AnswerSpecError as exc:
        return VerifierResult(
            ok=False,
            score=0.0,
            reason=f"invalid_predicted_integer:{exc}",
            expected_canonical=spec.canonical,
            verifier_id="numeric.integer",
        )


def verify_rational(predicted: str, spec: AnswerSpec) -> VerifierResult:
    try:
        pred = canonicalize_answer("rational", predicted)
        ok = Fraction(pred) == Fraction(spec.canonical)
        return VerifierResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="match" if ok else "rational_mismatch",
            predicted_canonical=pred,
            expected_canonical=spec.canonical,
            verifier_id="numeric.rational",
        )
    except (AnswerSpecError, ValueError, ZeroDivisionError) as exc:
        return VerifierResult(
            ok=False,
            score=0.0,
            reason=f"invalid_predicted_rational:{exc}",
            expected_canonical=spec.canonical,
            verifier_id="numeric.rational",
        )


def verify_decimal_exact(predicted: str, spec: AnswerSpec) -> VerifierResult:
    try:
        pred = canonicalize_answer("decimal_exact", predicted)
        ok = Fraction(pred) == Fraction(spec.canonical)
        return VerifierResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="match" if ok else "decimal_exact_mismatch",
            predicted_canonical=pred,
            expected_canonical=spec.canonical,
            verifier_id="numeric.decimal_exact",
        )
    except (AnswerSpecError, ValueError) as exc:
        return VerifierResult(
            ok=False,
            score=0.0,
            reason=f"invalid_predicted_decimal_exact:{exc}",
            expected_canonical=spec.canonical,
            verifier_id="numeric.decimal_exact",
        )


def verify_decimal_approx(predicted: str, spec: AnswerSpec) -> VerifierResult:
    tol = spec.tolerance if spec.tolerance is not None else 1e-6
    try:
        pred = float(canonicalize_answer("decimal_approx", predicted, tolerance=tol))
        exp = float(spec.canonical)
        ok = abs(pred - exp) <= tol
        return VerifierResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="match" if ok else "decimal_approx_mismatch",
            predicted_canonical=repr(pred),
            expected_canonical=spec.canonical,
            verifier_id="numeric.decimal_approx",
        )
    except (AnswerSpecError, ValueError) as exc:
        return VerifierResult(
            ok=False,
            score=0.0,
            reason=f"invalid_predicted_decimal_approx:{exc}",
            expected_canonical=spec.canonical,
            verifier_id="numeric.decimal_approx",
        )


def verify_logic_json(predicted: str, spec: AnswerSpec) -> VerifierResult:
    try:
        pred_canon = canonicalize_answer("logic_json", predicted)
        pred_obj = json.loads(pred_canon)
        exp_obj = (
            spec.ground_truth_structured
            if spec.ground_truth_structured is not None
            else json.loads(spec.canonical)
        )
        ok = _flat_logic_equal(pred_obj, exp_obj)
        return VerifierResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="match" if ok else "logic_json_mismatch",
            predicted_canonical=pred_canon,
            expected_canonical=spec.canonical,
            verifier_id="logic.flat_json",
        )
    except (AnswerSpecError, json.JSONDecodeError, TypeError) as exc:
        return VerifierResult(
            ok=False,
            score=0.0,
            reason=f"invalid_predicted_logic_json:{exc}",
            expected_canonical=spec.canonical,
            verifier_id="logic.flat_json",
        )


_REGISTRY: dict[str, Callable[[str, AnswerSpec], VerifierResult]] = {
    "integer": verify_integer,
    "rational": verify_rational,
    "decimal_exact": verify_decimal_exact,
    "decimal_approx": verify_decimal_approx,
    "logic_json": verify_logic_json,
}


def get_verifier(type_name: str) -> Callable[[str, AnswerSpec], VerifierResult]:
    reject_symbolic(type_name)
    key = str(type_name).strip().lower()
    if key not in _REGISTRY:
        raise AnswerSpecError(f"no verifier registered for type {type_name!r}")
    return _REGISTRY[key]


def verify_answer(
    predicted_or_completion: str,
    answer_spec: AnswerSpec | Mapping[str, Any],
    *,
    from_completion: bool = False,
) -> VerifierResult:
    """Verify a predicted answer (or full completion) against answer_spec."""
    spec = parse_answer_spec(answer_spec)
    if from_completion:
        extracted = extract_answer(predicted_or_completion)
        if extracted is None:
            return VerifierResult(
                ok=False,
                score=0.0,
                reason="missing_answer_block",
                expected_canonical=spec.canonical,
                verifier_id=f"registry.{spec.type}",
            )
        predicted = extracted
    else:
        predicted = predicted_or_completion
    return get_verifier(spec.type)(predicted, spec)


def verify_legacy(
    completion: str,
    ground_truth: Any,
    domain: str,
    puzzle_type: Optional[str] = None,
) -> float:
    """Compatibility shim used by reward composers for pre-answer_spec rows."""
    _ = puzzle_type
    from rlvr_contracts.answer_spec import infer_answer_spec_from_legacy

    try:
        if domain == "logic":
            # Preserve legacy set/list/dict semantics via logic_json when possible.
            if isinstance(ground_truth, (set, tuple)):
                ground_truth = list(ground_truth)
            if isinstance(ground_truth, str):
                try:
                    ground_truth = json.loads(ground_truth)
                except json.JSONDecodeError:
                    pass
            if not isinstance(ground_truth, (dict, list)):
                # Fall back to string equality for odd legacy logic GTs.
                pred = extract_answer(completion)
                if pred is None:
                    return 0.0
                try:
                    parsed = json.loads(pred)
                except json.JSONDecodeError:
                    parsed = pred
                return 1.0 if parsed == ground_truth else 0.0
            spec = parse_answer_spec({"type": "logic_json", "canonical": ground_truth})
        else:
            # Prefer numeric float equality for classic math legacy rows.
            pred = extract_answer(completion)
            if pred is None:
                return 0.0
            try:
                g_num = float(ground_truth)
                a_num = float(pred)
                return 1.0 if abs(g_num - a_num) < 1e-6 else 0.0
            except (TypeError, ValueError):
                try:
                    spec = infer_answer_spec_from_legacy(ground_truth, domain=domain)
                except AnswerSpecError:
                    return (
                        1.0
                        if str(pred).strip().lower() == str(ground_truth).strip().lower()
                        else 0.0
                    )
                return verify_answer(pred, spec).score
        return verify_answer(completion, spec, from_completion=True).score
    except AnswerSpecError:
        return 0.0
