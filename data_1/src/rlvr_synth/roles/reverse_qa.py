"""Reverse-QA: answer-first question generation with verifier gate.

Canonical 2026 contract:
  1. Own the label (programmatic solver GT + solution_steps).
  2. LLM writes an Arabic question whose answer is exactly that GT
     (optionally without using the template prompt).
  3. Keep only if independent checks recover the same GT; else fall back
     to the template prompt (or reject when no template).

Modes:
  - ``paraphrase``: rewrite existing template wording (legacy add-on).
  - ``full``: generate question from (GT, solution_steps, latent) only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any, Callable

from rlvr_synth.roles.protocols import RenderedProblem

# (domain, answer_spec, latent) -> Arabic question text
GenerateFn = Callable[[str, dict[str, Any], dict[str, Any]], str]
# (prompt, domain) -> model answer text / structured
SolveFn = Callable[[str, str], str]
# legacy paraphrase signature
RewriteFn = Callable[[str, str, dict[str, Any], dict[str, Any]], str]

_DIGIT_RE = re.compile(r"\d+")
_THINK_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)


def arabic_char_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or ("\u0600" <= c <= "\u06FF")]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return ar / max(len(letters), 1)


def _canonical_strings(answer_spec: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("canonical", "ground_truth_structured", "ground_truth"):
        val = answer_spec.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            out.append(json.dumps(val, ensure_ascii=False, sort_keys=True))
            out.append(json.dumps(val, ensure_ascii=False))
        else:
            s = str(val).strip()
            if s:
                out.append(s)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _primary_canonical(answer_spec: dict[str, Any]) -> str:
    cans = _canonical_strings(answer_spec)
    return cans[0] if cans else ""


def latent_operand_digits(latent: dict[str, Any], answer_spec: dict[str, Any]) -> set[str]:
    """Digits that should remain in a math question (operands), excluding GT."""
    blob_parts: list[str] = []
    for key in ("solution_steps", "meta_extra", "prompt"):
        val = latent.get(key)
        if isinstance(val, list):
            blob_parts.extend(str(x) for x in val)
        elif isinstance(val, dict):
            blob_parts.append(json.dumps(val, ensure_ascii=False))
        elif val is not None:
            blob_parts.append(str(val))
    # Common numeric fields
    for key, val in (latent or {}).items():
        if isinstance(val, (int, float)) and key not in {"seed", "num_steps", "ground_truth"}:
            blob_parts.append(str(val))
        if key in {"a", "b", "c", "n", "k", "rate", "price", "qty"}:
            blob_parts.append(str(val))
    digits = set(_DIGIT_RE.findall(" ".join(blob_parts)))
    gt = {_DIGIT_RE.findall(c)[0] for c in _canonical_strings(answer_spec) if _DIGIT_RE.findall(c)}
    # Prefer operands that are not the final answer token
    ops = digits - gt
    return ops or digits


def extract_answer_text(raw: str) -> str:
    text = (raw or "").strip()
    m = _THINK_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # JSON {"answer": ...}
    jm = re.search(r"\{[\s\S]*\}", text)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            if isinstance(obj, dict):
                for key in ("answer", "canonical", "final"):
                    if key in obj and obj[key] is not None:
                        v = obj[key]
                        if isinstance(v, (dict, list)):
                            return json.dumps(v, ensure_ascii=False, sort_keys=True)
                        return str(v).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[-1].strip() if text else ""


def answers_match(predicted: str, answer_spec: dict[str, Any]) -> bool:
    pred = (predicted or "").strip()
    if not pred:
        return False
    # Strip common wrappers
    pred = pred.strip("`\"' ")
    for canon in _canonical_strings(answer_spec):
        if pred == canon:
            return True
        # numeric tolerance for ints
        try:
            if float(pred) == float(canon):
                return True
        except (TypeError, ValueError):
            pass
        if canon in pred and len(canon) >= 1:
            # letter answers
            if len(canon) <= 3 and re.search(rf"(?<![A-Za-z0-9]){re.escape(canon)}(?![A-Za-z0-9])", pred):
                return True
    # logic_json: try parse
    if (answer_spec or {}).get("type") == "logic_json":
        try:
            from rlvr_contracts.verifiers import verify_answer

            fake = f"<think>تحقق</think>\n<answer>{pred}</answer>"
            return bool(verify_answer(fake, answer_spec, from_completion=True).ok)
        except Exception:
            return False
    return False


def validate_generated_question(
    *,
    prompt: str,
    answer_spec: dict[str, Any],
    domain: str,
    latent: dict[str, Any] | None = None,
    template_prompt: str | None = None,
    min_arabic_ratio: float = 0.55,
    min_len: int = 12,
    require_operands: bool = True,
) -> tuple[bool, str]:
    """Static gates before optional re-solve."""
    new = (prompt or "").strip()
    if not new:
        return False, "empty_rewrite"
    if len(new) < min_len:
        return False, "too_short"
    if arabic_char_ratio(new) < min_arabic_ratio:
        return False, "low_arabic"
    if template_prompt and new.strip() == template_prompt.strip():
        return False, "unchanged"

    for canon in _canonical_strings(answer_spec):
        if not canon:
            continue
        if re.search(
            rf"(الإجابة|الناتج|الجواب|correct answer)\s*[:=]?\s*{re.escape(canon)}",
            new,
            flags=re.I,
        ):
            return False, "gt_leak_phrase"
        # Full mode: never allow GT token in the question when it wasn't an operand-only case
        if len(canon) >= 1:
            if len(canon) >= 6 and canon in new:
                return False, "gt_leak"
            if len(canon) < 6 and re.search(rf"(?<!\d){re.escape(canon)}(?!\d)", new):
                # Allow only if this digit is a required operand (e.g. answer equals an operand — rare)
                ops = latent_operand_digits(latent or {}, answer_spec)
                if canon not in ops:
                    return False, "gt_leak"

    if require_operands and domain in {"gsm8k", "math", "math_comp"}:
        ops = latent_operand_digits(latent or {}, answer_spec)
        # Prefer template digits minus GT when latent sparse
        if not ops and template_prompt:
            ops = set(_DIGIT_RE.findall(template_prompt)) - {
                d for c in _canonical_strings(answer_spec) for d in _DIGIT_RE.findall(c)
            }
        if ops:
            present = set(_DIGIT_RE.findall(new))
            if not (ops & present):
                return False, "lost_operands"
    return True, "ok"


# Back-compat name used by older tests
def validate_reverse_qa(
    *,
    original_prompt: str,
    rewritten_prompt: str,
    answer_spec: dict[str, Any],
    domain: str,
    min_arabic_ratio: float = 0.55,
    min_len: int = 12,
) -> tuple[bool, str]:
    return validate_generated_question(
        prompt=rewritten_prompt,
        answer_spec=answer_spec,
        domain=domain,
        latent={"prompt": original_prompt},
        template_prompt=original_prompt,
        min_arabic_ratio=min_arabic_ratio,
        min_len=min_len,
        require_operands=True,
    )


def _extract_json_prompt(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for key in ("prompt", "question", "rewritten_prompt", "arabic_prompt"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        except json.JSONDecodeError:
            pass
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_full_reverse_qa_messages(
    *,
    domain: str,
    answer_spec: dict[str, Any],
    latent: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    latent = dict(latent or {})
    steps = latent.get("solution_steps") or []
    meta = latent.get("meta_extra") or {}
    # Do not pass the template prompt — answer-first.
    payload = {
        "task": "reverse_qa_full",
        "domain": domain,
        "fixed_answer": {
            "type": answer_spec.get("type"),
            "canonical": answer_spec.get("canonical"),
        },
        "solution_steps": steps[:12],
        "structure_hints": {
            k: meta.get(k)
            for k in ("template_family", "topic", "num_steps")
            if k in meta or k in latent
        },
        "num_steps": latent.get("num_steps"),
        "template_family": latent.get("template_family") or meta.get("template_family"),
        "instructions": [
            "Write a NEW Formal Arabic word problem.",
            "The unique correct final answer MUST be exactly fixed_answer.canonical.",
            "Use solution_steps as the intended reasoning path.",
            "Do NOT include the final answer value anywhere in the question.",
            "Do NOT copy an English template; write natural Arabic.",
            "Output JSON only: {\"prompt\": \"...\"}",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You perform Reverse-QA for RLVR: invent Arabic question wording "
                "around a FIXED verified answer you must not reveal."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_reverse_qa_messages(
    *,
    original_prompt: str,
    domain: str,
    answer_spec: dict[str, Any],
    latent: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Paraphrase-mode messages (legacy)."""
    latent = latent or {}
    steps = latent.get("solution_steps") or []
    payload = {
        "domain": domain,
        "original_prompt": original_prompt,
        "fixed_answer_spec": {
            "type": answer_spec.get("type"),
            "canonical": answer_spec.get("canonical"),
        },
        "solution_steps": steps[:8],
        "instructions": [
            "Rewrite ONLY the Arabic problem statement (surface form).",
            "Keep the SAME solvable problem and the SAME final answer.",
            "Do NOT include the final answer value in the question text.",
            "Do NOT add new numbers that change the math.",
            "Keep formal Arabic; output JSON: {\"prompt\": \"...\"}",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an Arabic problem rewriter for RLVR reverse-QA. "
                "Diversify wording while preserving the fixed ground-truth answer. "
                "Never put the answer in the question."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_resolve_messages(*, prompt: str, domain: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Solve the Arabic problem. Reply with JSON "
                '{"answer": "<final answer only>"} and no explanation.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"domain": domain, "prompt": prompt},
                ensure_ascii=False,
            ),
        },
    ]


def _openai_client_and_model(model: str | None = None):
    from openai import OpenAI
    from synth.dspy_teacher import _resolve_lm_endpoint

    use_model = model or os.environ.get("REVERSE_QA_MODEL") or os.environ.get(
        "TEACHER_MODEL", "deepseek-v4-flash"
    )
    dspy_model, api_key, api_base = _resolve_lm_endpoint(use_model)
    api_model = dspy_model.split("/", 1)[-1] if "/" in dspy_model else dspy_model
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client, api_model


def make_openai_compatible_generate_fn(
    *,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 900,
) -> GenerateFn:
    def _generate(domain: str, answer_spec: dict[str, Any], latent: dict[str, Any]) -> str:
        client, api_model = _openai_client_and_model(model)
        messages = build_full_reverse_qa_messages(
            domain=domain, answer_spec=answer_spec, latent=latent
        )
        resp = client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        return _extract_json_prompt(content)

    return _generate


def make_openai_compatible_solve_fn(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> SolveFn:
    def _solve(prompt: str, domain: str) -> str:
        client, api_model = _openai_client_and_model(model)
        messages = build_resolve_messages(prompt=prompt, domain=domain)
        resp = client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        return extract_answer_text(content)

    return _solve


def make_openai_compatible_rewrite_fn(
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> RewriteFn:
    def _rewrite(
        original_prompt: str,
        domain: str,
        answer_spec: dict[str, Any],
        latent: dict[str, Any],
    ) -> str:
        client, api_model = _openai_client_and_model(model)
        messages = build_reverse_qa_messages(
            original_prompt=original_prompt,
            domain=domain,
            answer_spec=answer_spec,
            latent=latent,
        )
        resp = client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        return _extract_json_prompt(content)

    return _rewrite


def reverse_qa_mode(config: dict[str, Any] | None = None) -> str:
    cfg = config or {}
    if cfg.get("reverse_qa_mode"):
        return str(cfg["reverse_qa_mode"]).strip().lower()
    return os.environ.get("REVERSE_QA_MODE", "full").strip().lower() or "full"


def reverse_qa_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    if "reverse_qa" in cfg:
        return bool(cfg.get("reverse_qa"))
    return os.environ.get("REVERSE_QA", "0").strip() in {"1", "true", "True", "yes"}


def reverse_qa_resolve_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    if "reverse_qa_resolve" in cfg:
        return bool(cfg.get("reverse_qa_resolve"))
    return os.environ.get("REVERSE_QA_RESOLVE", "1").strip() in {"1", "true", "True", "yes"}


def diversify_rendered_problem(
    problem: RenderedProblem,
    *,
    rewrite_fn: RewriteFn | None = None,
    generate_fn: GenerateFn | None = None,
    solve_fn: SolveFn | None = None,
    latent: dict[str, Any] | None = None,
    enabled: bool = True,
    mode: str = "full",
    require_resolve: bool = True,
) -> RenderedProblem:
    """Answer-first Reverse-QA (full) or paraphrase; always freezes answer_spec."""
    if not enabled:
        return problem

    template = problem.prompt
    latent = dict(latent or problem.metadata or {})
    spec = dict(problem.answer_spec or {})
    mode = (mode or "full").lower()

    def _fail(reason: str) -> RenderedProblem:
        meta = dict(problem.metadata or {})
        meta["reverse_qa"] = {
            "applied": False,
            "reason": reason,
            "mode": mode,
            "original_prompt": template,
        }
        return replace(
            problem,
            metadata=meta,
            provenance={**dict(problem.provenance or {}), "reverse_qa": False},
        )

    try:
        if mode == "full":
            if generate_fn is None:
                return _fail("missing_generate_fn")
            raw = generate_fn(problem.domain, spec, latent)
        else:
            if rewrite_fn is None:
                return _fail("missing_rewrite_fn")
            raw = rewrite_fn(template, problem.domain, spec, latent)
        raw = _extract_json_prompt(str(raw or ""))
    except Exception as exc:
        return _fail(f"rewrite_error:{type(exc).__name__}")

    ok, reason = validate_generated_question(
        prompt=raw,
        answer_spec=spec,
        domain=problem.domain,
        latent=latent,
        template_prompt=template if mode != "full" else None,
        require_operands=True,
    )
    if not ok:
        return _fail(reason)

    if require_resolve:
        if solve_fn is None:
            return _fail("missing_solve_fn")
        try:
            predicted = solve_fn(raw, problem.domain)
        except Exception as exc:
            return _fail(f"resolve_error:{type(exc).__name__}")
        if not answers_match(predicted, spec):
            return _fail("resolve_mismatch")

    from rlvr_synth.quality.ids import content_problem_id

    new_pid = content_problem_id(
        domain=problem.domain,
        prompt=raw.strip(),
        answer_canonical=spec.get("canonical"),
        partition=problem.partition,
        seed=int((problem.provenance or {}).get("seed") or 0),
    )
    meta = dict(problem.metadata or {})
    meta["reverse_qa"] = {
        "applied": True,
        "reason": "ok",
        "mode": mode,
        "original_prompt": template,
        "resolved": bool(require_resolve),
    }
    return replace(
        problem,
        problem_id=new_pid,
        prompt=raw.strip(),
        answer_spec=spec,
        metadata=meta,
        provenance={
            **dict(problem.provenance or {}),
            "reverse_qa": True,
            "reverse_qa_mode": mode,
            "source": f"programmatic+reverse_qa_{mode}",
        },
    )
