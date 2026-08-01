"""Discriminated answer_spec canonicalization.

Supported types: integer, rational, decimal_exact, decimal_approx, logic_json,
mcq_letter, constraint_set.
Symbolic answers are explicitly rejected at every boundary.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Mapping, Optional

SUPPORTED_ANSWER_TYPES = frozenset(
    {
        "integer",
        "rational",
        "decimal_exact",
        "decimal_approx",
        "logic_json",
        "mcq_letter",
        "constraint_set",
    }
)
REJECTED_ANSWER_TYPES = frozenset({"symbolic", "sympy", "expression"})

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_FRAC_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
_INT_RE = re.compile(r"^\s*-?\d+\s*$")
_DECIMAL_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
_MCQ_LETTER_RE = re.compile(r"^\s*([A-Da-d]|[أاببججدد])\s*[).:\-،,]?\s*$")
_MCQ_LETTER_MAP = {
    "أ": "A",
    "ا": "A",
    "ب": "B",
    "ج": "C",
    "د": "D",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}


class AnswerSpecError(ValueError):
    """Raised when an answer_spec is unsupported or malformed."""


@dataclass(frozen=True)
class AnswerSpec:
    type: str
    canonical: str
    tolerance: Optional[float] = None
    verifier_artifact: Optional[str] = None
    ground_truth_structured: Any = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "canonical": self.canonical,
        }
        if self.tolerance is not None:
            out["tolerance"] = self.tolerance
        if self.verifier_artifact is not None:
            out["verifier_artifact"] = self.verifier_artifact
        if self.ground_truth_structured is not None:
            out["ground_truth_structured"] = self.ground_truth_structured
        if self.extras:
            out.update(dict(self.extras))
        return out


def reject_symbolic(type_name: str | None) -> None:
    """Fail closed on symbolic / expression answer types (v1 policy)."""
    if type_name is None:
        return
    key = str(type_name).strip().lower()
    if key in REJECTED_ANSWER_TYPES or key.startswith("symbolic"):
        raise AnswerSpecError(
            f"answer type '{type_name}' is rejected; "
            "use integer|rational|decimal_exact|decimal_approx|logic_json|mcq_letter|constraint_set"
        )


def _normalize_numeric_text(raw: str) -> str:
    text = str(raw).strip().translate(_ARABIC_DIGITS)
    text = text.replace(",", "").replace("٬", "")
    text = text.replace("−", "-").replace("–", "-")
    # Strip trailing percentage only when explicit percent form is requested upstream.
    return text.strip()


def canonicalize_integer(value: Any) -> str:
    if isinstance(value, bool):
        raise AnswerSpecError("boolean is not a valid integer answer")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value - round(value)) > 1e-9:
            raise AnswerSpecError(f"non-integral float for integer: {value}")
        return str(int(round(value)))
    text = _normalize_numeric_text(str(value))
    if "/" in text:
        frac = Fraction(text)
        if frac.denominator != 1:
            raise AnswerSpecError(f"non-integer rational: {text}")
        return str(frac.numerator)
    if not _INT_RE.match(text):
        # Allow 3.0 → 3
        if _DECIMAL_RE.match(text):
            f = float(text)
            if abs(f - round(f)) <= 1e-9:
                return str(int(round(f)))
        raise AnswerSpecError(f"invalid integer: {value!r}")
    return str(int(text))


def canonicalize_rational(value: Any) -> str:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value}/1"
    if isinstance(value, float):
        frac = Fraction(value).limit_denominator(10_000)
        return f"{frac.numerator}/{frac.denominator}"
    text = _normalize_numeric_text(str(value))
    if _FRAC_RE.match(text):
        frac = Fraction(text)
        return f"{frac.numerator}/{frac.denominator}"
    if _INT_RE.match(text):
        return f"{int(text)}/1"
    if _DECIMAL_RE.match(text):
        frac = Fraction(text).limit_denominator(10_000)
        return f"{frac.numerator}/{frac.denominator}"
    raise AnswerSpecError(f"invalid rational: {value!r}")


def canonicalize_decimal_exact(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise AnswerSpecError(f"non-finite decimal: {value}")
        # `format(float, "f")` defaults to six decimals and silently turns
        # values such as 1e-7 into zero. Decimal(str(...)) preserves the
        # user-visible float value while rendering without exponent notation.
        text = format(Decimal(str(value)), "f")
    else:
        text = _normalize_numeric_text(str(value))
        if not _DECIMAL_RE.match(text) and not _INT_RE.match(text):
            raise AnswerSpecError(f"invalid decimal_exact: {value!r}")
    # Normalize trailing zeros but keep significant fraction digits.
    if "." in text:
        text = text.rstrip("0").rstrip(".") if text.split(".", 1)[1] else text
    return text or "0"


def canonicalize_decimal_approx(value: Any, tolerance: float | None = None) -> str:
    # Store a stable float rendering; comparison uses tolerance.
    _ = tolerance
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise AnswerSpecError(f"non-finite decimal_approx: {value}")
        return repr(float(value))
    text = _normalize_numeric_text(str(value))
    try:
        return repr(float(text))
    except ValueError as exc:
        raise AnswerSpecError(f"invalid decimal_approx: {value!r}") from exc


def canonicalize_logic_json(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnswerSpecError(f"logic_json must be valid JSON: {value!r}") from exc
        if not isinstance(parsed, (dict, list)):
            raise AnswerSpecError("logic_json must be a JSON object or array")
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raise AnswerSpecError(f"invalid logic_json: {type(value).__name__}")


def canonicalize_mcq_letter(value: Any) -> str:
    text = str(value).strip()
    # Accept "C" / "ج" / "الخيار C" / "الإجابة: B"
    m = re.search(r"([A-Da-d]|[أاببججدد])\b", text)
    if not m:
        m = _MCQ_LETTER_RE.match(text)
        if not m:
            raise AnswerSpecError(f"invalid mcq_letter: {value!r}")
        token = m.group(1)
    else:
        token = m.group(1)
    letter = _MCQ_LETTER_MAP.get(token, token.upper())
    if letter not in {"A", "B", "C", "D"}:
        raise AnswerSpecError(f"mcq_letter out of range: {value!r}")
    return letter


def canonicalize_constraint_set(value: Any) -> str:
    """Canonicalize a constraint_set answer_spec payload.

    Expected structured form::
        {"constraints": [{"id": "...", "category": "...", "params": {...}}, ...],
         "pass_all": true}
    ``canonical`` is a stable JSON digest used for equality checks of the spec
    itself; verification of completions uses the structured constraints list.
    """
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnswerSpecError(
                f"constraint_set must be valid JSON: {value!r}"
            ) from exc
    elif isinstance(value, dict):
        parsed = value
    else:
        raise AnswerSpecError(f"invalid constraint_set: {type(value).__name__}")
    constraints = parsed.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise AnswerSpecError("constraint_set.constraints must be a non-empty list")
    normalized = []
    for item in constraints:
        if not isinstance(item, dict):
            raise AnswerSpecError("each constraint must be an object")
        cid = str(item.get("id") or item.get("category") or "").strip()
        category = str(item.get("category") or "").strip()
        if not cid or not category:
            raise AnswerSpecError("constraint id and category are required")
        normalized.append(
            {
                "id": cid,
                "category": category,
                "params": dict(item.get("params") or {}),
            }
        )
    normalized.sort(key=lambda c: c["id"])
    payload = {
        "constraints": normalized,
        "pass_all": bool(parsed.get("pass_all", True)),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonicalize_answer(type_name: str, value: Any, *, tolerance: float | None = None) -> str:
    reject_symbolic(type_name)
    key = str(type_name).strip().lower()
    if key not in SUPPORTED_ANSWER_TYPES:
        raise AnswerSpecError(f"unsupported answer type: {type_name!r}")
    if key == "integer":
        return canonicalize_integer(value)
    if key == "rational":
        return canonicalize_rational(value)
    if key == "decimal_exact":
        return canonicalize_decimal_exact(value)
    if key == "decimal_approx":
        return canonicalize_decimal_approx(value, tolerance=tolerance)
    if key == "logic_json":
        return canonicalize_logic_json(value)
    if key == "mcq_letter":
        return canonicalize_mcq_letter(value)
    if key == "constraint_set":
        return canonicalize_constraint_set(value)
    raise AnswerSpecError(f"unsupported answer type: {type_name!r}")


def parse_answer_spec(raw: Mapping[str, Any] | AnswerSpec | None) -> AnswerSpec:
    if raw is None:
        raise AnswerSpecError("answer_spec is required")
    if isinstance(raw, AnswerSpec):
        reject_symbolic(raw.type)
        if raw.type not in SUPPORTED_ANSWER_TYPES:
            raise AnswerSpecError(f"unsupported answer type: {raw.type!r}")
        return raw
    if not isinstance(raw, Mapping):
        raise AnswerSpecError(f"answer_spec must be a mapping, got {type(raw).__name__}")

    type_name = raw.get("type")
    if type_name is None:
        raise AnswerSpecError("answer_spec.type is required")
    reject_symbolic(str(type_name))
    key = str(type_name).strip().lower()
    if key not in SUPPORTED_ANSWER_TYPES:
        raise AnswerSpecError(f"unsupported answer type: {type_name!r}")

    tolerance = raw.get("tolerance")
    if key == "decimal_approx" and tolerance is None:
        tolerance = 1e-6
    if tolerance is not None:
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise AnswerSpecError("answer_spec.tolerance must be numeric") from exc
        if not math.isfinite(tolerance) or tolerance < 0:
            raise AnswerSpecError(
                "answer_spec.tolerance must be finite and non-negative"
            )
        if key != "decimal_approx":
            raise AnswerSpecError(
                "answer_spec.tolerance is only valid for decimal_approx"
            )

    canonical_in = raw.get("canonical", raw.get("value", raw.get("ground_truth")))
    if canonical_in is None and "ground_truth_structured" in raw:
        canonical_in = raw["ground_truth_structured"]
    if canonical_in is None:
        raise AnswerSpecError("answer_spec.canonical is required")

    canonical = canonicalize_answer(key, canonical_in, tolerance=tolerance)
    structured = raw.get("ground_truth_structured")
    if structured is None and key == "logic_json":
        structured = json.loads(canonical)
    elif structured is None and key == "constraint_set":
        structured = json.loads(canonical)
    elif structured is None and key == "rational":
        num, den = canonical.split("/", 1)
        structured = {"numerator": int(num), "denominator": int(den)}
    elif structured is None and key == "integer":
        structured = int(canonical)
    elif structured is None and key.startswith("decimal"):
        structured = float(canonical)

    known = {
        "type",
        "canonical",
        "value",
        "ground_truth",
        "tolerance",
        "verifier_artifact",
        "ground_truth_structured",
    }
    extras = {k: v for k, v in raw.items() if k not in known}
    return AnswerSpec(
        type=key,
        canonical=canonical,
        tolerance=float(tolerance) if tolerance is not None else None,
        verifier_artifact=raw.get("verifier_artifact"),
        ground_truth_structured=structured,
        extras=extras,
    )


def infer_answer_spec_from_legacy(ground_truth: Any, domain: str = "math") -> AnswerSpec:
    """Best-effort adapter for legacy v2 rows without answer_spec."""
    if str(domain).lower() == "logic":
        return parse_answer_spec({"type": "logic_json", "canonical": ground_truth})
    if isinstance(ground_truth, bool):
        raise AnswerSpecError("boolean ground truth is not supported without answer_spec")
    if isinstance(ground_truth, int):
        return parse_answer_spec({"type": "integer", "canonical": ground_truth})
    if isinstance(ground_truth, float):
        return parse_answer_spec(
            {"type": "decimal_approx", "canonical": ground_truth, "tolerance": 1e-6}
        )
    text = str(ground_truth).strip()
    # Strip optional variable assignment prefix e.g. "x = 26" -> "26"
    eq_match = re.match(r"^[a-zA-Z\u0600-\u06ff\s]+=+\s*(-?\d+(?:\.\d+)?)$", text)
    if eq_match:
        text = eq_match.group(1).strip()
    if _FRAC_RE.match(text.translate(_ARABIC_DIGITS)):
        return parse_answer_spec({"type": "rational", "canonical": text})
    if _INT_RE.match(text.translate(_ARABIC_DIGITS).replace(",", "")):
        return parse_answer_spec({"type": "integer", "canonical": text})
    if _DECIMAL_RE.match(text.translate(_ARABIC_DIGITS).replace(",", "")):
        return parse_answer_spec(
            {"type": "decimal_approx", "canonical": text, "tolerance": 1e-6}
        )
    # Legacy free-text fallback treated as logic_json string wrapper is not allowed;
    # raise so loaders can quarantine.
    raise AnswerSpecError(
        f"cannot infer supported answer_spec from legacy ground truth: {ground_truth!r}"
    )
