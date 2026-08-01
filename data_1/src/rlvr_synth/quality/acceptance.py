"""Per-row acceptance gates for V4 SFT and RLVR releases.

Never trust stored verifier_result / trace_audit: always replay.
"""

from __future__ import annotations

from typing import Any

from rlvr_contracts.answer_spec import AnswerSpecError, parse_answer_spec, reject_symbolic
from rlvr_contracts.leak import leak_policy_score, structural_leak_score
from rlvr_contracts.response import extract_think, parse_response
from rlvr_contracts.schema_validate import validate_record
from rlvr_contracts.verifiers import verify_answer
from rlvr_synth.qa.arabic_metrics import score_arabic_record
from rlvr_synth.quality.quotas import CORE_DOMAINS

SUPPORTED_ANSWER_TYPES = {
    "integer",
    "rational",
    "decimal_exact",
    "decimal_approx",
    "logic_json",
}

try:
    from synth.dspy_teacher import (
        think_has_bare_ops,
        think_has_answer_equation,
        think_is_incomplete,
        think_is_vague_narration,
        think_quality_ok,
    )
except Exception:  # pragma: no cover - vendor optional in some installs
    think_has_bare_ops = None  # type: ignore
    think_has_answer_equation = None  # type: ignore
    think_is_incomplete = None  # type: ignore
    think_is_vague_narration = None  # type: ignore
    think_quality_ok = None  # type: ignore


def _canonical_from_spec(spec: dict[str, Any] | None) -> Any:
    if not spec:
        return None
    return spec.get("ground_truth_structured", spec.get("canonical"))


def _oracle_matches(row: dict[str, Any], spec: Any) -> tuple[bool, str]:
    """Compare stored answer_spec against latent/programmatic ground truth when present."""
    meta = row.get("metadata") or {}
    latent = (row.get("lineage") or {}).get("latent") or meta.get("latent") or {}
    gt = (
        meta.get("ground_truth")
        or meta.get("ground_truth_answer")
        or latent.get("ground_truth")
        or row.get("oracle_ground_truth")
    )
    if gt is None:
        return True, "no_oracle_present"
    try:
        from rlvr_contracts.answer_spec import infer_answer_spec_from_legacy

        domain = str(row.get("domain") or "")
        reward_domain = "logic" if domain == "logic" else "math"
        expected = infer_answer_spec_from_legacy(gt, domain=reward_domain)
        got = spec if hasattr(spec, "canonical") else parse_answer_spec(row["answer_spec"])
        if expected.type != got.type:
            return False, f"oracle_type_mismatch:{expected.type}!={got.type}"
        if str(expected.canonical) != str(got.canonical):
            # Allow numeric equality for ints/rationals encoded differently.
            try:
                if float(expected.canonical) == float(got.canonical):
                    return True, "oracle_numeric_equal"
            except Exception:
                pass
            return False, "oracle_canonical_mismatch"
        return True, "oracle_match"
    except Exception as exc:
        return False, f"oracle_error:{exc}"


def replay_verifier(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        reject_symbolic(str((row.get("answer_spec") or {}).get("type", "")))
        spec = parse_answer_spec(row["answer_spec"])
    except (AnswerSpecError, KeyError, TypeError) as exc:
        return False, [f"answer_spec:{exc}"]

    domain = str(row.get("domain") or "")
    if domain == "logic" and spec.type != "logic_json":
        return False, ["logic_must_be_logic_json"]
    if domain and domain not in CORE_DOMAINS:
        return False, [f"non_core_domain:{domain}"]
    if spec.type not in SUPPORTED_ANSWER_TYPES:
        return False, [f"unsupported_answer_type:{spec.type}"]

    oracle_ok, oracle_reason = _oracle_matches(row, spec)
    if not oracle_ok:
        reasons.append(oracle_reason)
        return False, reasons

    response = str(row.get("response") or "")
    if response:
        parsed = parse_response(response)
        if not parsed.format_ok:
            return False, list(parsed.errors) or ["format_fail"]
        vr = verify_answer(response, spec, from_completion=True)
        if not vr.ok:
            return False, [vr.reason or "verifier_fail"]
    return True, reasons or ["ok"]


def _sft_reasoning_gates(row: dict[str, Any], response: str) -> list[str]:
    reasons: list[str] = []
    think = extract_think(response) or ""
    tokens = think.split()
    if len(tokens) < 20:
        reasons.append("think_too_short")
    if len(tokens) > 640:
        reasons.append("think_too_long")

    if leak_policy_score(response) != 0.0:
        reasons.append("leak_policy")
    if structural_leak_score(response) != 0.0:
        reasons.append("structural_leak")

    ar = score_arabic_record(row)
    if not ar.get("pass"):
        reasons.append("arabic_record_fail")

    domain = str(row.get("domain") or "")
    gt = _canonical_from_spec(row.get("answer_spec"))
    if think_quality_ok is not None:
        ok, tag = think_quality_ok(think, response, domain=domain or "gsm8k", gt=gt)
        if not ok:
            reasons.append(f"think_quality:{tag}")
    if think_is_incomplete is not None and think_is_incomplete(think):
        reasons.append("incomplete")
    if think_is_vague_narration is not None and think_is_vague_narration(think, domain):
        reasons.append("vague_narration")
    if domain != "logic":
        if think_has_bare_ops is not None and think_has_bare_ops(think):
            reasons.append("bare_op")
        if think_has_answer_equation is not None and gt is not None:
            if not think_has_answer_equation(think, gt, domain):
                reasons.append("missing_answer_eq")
    return reasons


def accept_sft_row(row: dict[str, Any], *, require_decontam_clean: bool = True) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    schema_errs = validate_record("sft_trace", row)
    if schema_errs:
        return False, [f"schema:{e}" for e in schema_errs[:5]]

    if row.get("partition") != "sft_train":
        reasons.append("bad_partition")

    ok, vr_reasons = replay_verifier(row)
    if not ok:
        reasons.extend(vr_reasons)

    response = str(row.get("response") or "")
    if not response:
        reasons.append("missing_response")
    else:
        reasons.extend(_sft_reasoning_gates(row, response))

    # Fresh audit flags only — never trust legacy teacher-only stubs.
    audit = row.get("trace_audit") or {}
    if audit.get("format_ok") is not True:
        # Recompute format independently; stored false/missing fails unless format_ok after replay.
        parsed = parse_response(response) if response else None
        if parsed is None or not parsed.format_ok:
            reasons.append("trace_audit_format_ok")
    if audit.get("step_verified") is not True:
        # Require caller to have written step_verified from live replay.
        reasons.append("trace_audit_step_verified_missing")

    if row.get("verifier_result") is not True:
        reasons.append("verifier_result_not_true_boolean")

    decontam = row.get("decontam") or (row.get("metadata") or {}).get("decontam") or {}
    status = decontam.get("status")
    if require_decontam_clean:
        if status != "clean":
            reasons.append(f"decontam:{status or 'missing'}")

    domain = str(row.get("domain") or "")
    if domain not in CORE_DOMAINS:
        reasons.append(f"non_core_domain:{domain}")

    return (len(reasons) == 0), reasons


def accept_rlvr_row(
    row: dict[str, Any],
    *,
    require_decontam_clean: bool = True,
    min_arabic: float = 0.80,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    schema_errs = validate_record("rlvr_prompt", row)
    if schema_errs:
        return False, [f"schema:{e}" for e in schema_errs[:5]]

    if row.get("partition") != "rlvr_train":
        reasons.append("bad_partition")
    if str(row.get("response") or "") != "":
        reasons.append("response_must_be_empty")

    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        reasons.append("empty_prompt")

    ok, vr_reasons = replay_verifier(row)
    if not ok:
        reasons.extend(vr_reasons)

    ar = score_arabic_record(row, calibration={"min_ratio": min_arabic})
    if float(ar.get("arabic_ratio", 0.0)) < min_arabic or not ar.get("pass"):
        # For prompts, dialect/foreign still fail; allow slightly lower ratio floor via min_arabic.
        if float(ar.get("arabic_ratio", 0.0)) < min_arabic:
            reasons.append("arabic_ratio_low")
        if ar.get("foreign_prose") or ar.get("dialect"):
            reasons.append("arabic_prompt_quality")

    decontam = row.get("decontam") or (row.get("metadata") or {}).get("decontam") or {}
    status = decontam.get("status")
    if require_decontam_clean and status != "clean":
        reasons.append(f"decontam:{status or 'missing'}")

    domain = str(row.get("domain") or "")
    if domain not in CORE_DOMAINS:
        reasons.append(f"non_core_domain:{domain}")

    for key in ("problem_id", "family_id", "verifier_version", "verifier_type"):
        if not row.get(key):
            reasons.append(f"missing:{key}")

    return (len(reasons) == 0), reasons


def stamp_fresh_sft_audit(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite verifier_result/trace_audit from live replay only."""
    out = dict(row)
    response = str(out.get("response") or "")
    parsed = parse_response(response) if response else None
    ok, reasons = replay_verifier(out)
    think = (parsed.think if parsed else "") or ""
    out["verifier_result"] = bool(ok and parsed and parsed.format_ok)
    method_id = (out.get("trace_audit") or {}).get("method_id") or "dspy_0"
    out["trace_audit"] = {
        "format_ok": bool(parsed.format_ok) if parsed else False,
        "step_verified": bool(ok and parsed and parsed.format_ok and len(think.split()) >= 20),
        "concision_tokens": len(think.split()),
        "method_id": str(method_id),
        "rejection_reasons": [] if ok else list(reasons),
        "replay_source": "accept_sft_row",
    }
    if out.get("partition") in {"sft", "train", ""}:
        out["partition"] = "sft_train"
    return out
