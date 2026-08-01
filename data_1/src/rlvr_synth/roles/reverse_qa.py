"""Reverse-QA prompt diversification.

Pipeline contract (2026 baseline + frontier add-on):
  1. Programmatic template generates Arabic prompt + solver GT (unchanged).
  2. LLM rewrites ONLY the surface Arabic problem statement.
  3. ``answer_spec`` / oracle GT stay fixed.
  4. Verifier + leak gates accept or fall back to the template prompt.

Family IDs are assigned before this step (orchestrator), so diversification
does not break SFT↔RLVR family isolation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any, Callable, Optional

from rlvr_synth.roles.protocols import RenderedProblem

RewriteFn = Callable[[str, str, dict[str, Any], dict[str, Any]], str]

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_DIGIT_RE = re.compile(r"\d+")


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
    # Dedup preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def gt_leaks_in_prompt(prompt: str, answer_spec: dict[str, Any]) -> bool:
    """True if the fixed answer appears as a standalone leak in the question."""
    text = prompt or ""
    for canon in _canonical_strings(answer_spec):
        if not canon:
            continue
        if len(canon) >= 8 and canon in text:
            return True
        # Numeric / short answers: require token boundary style match
        if re.search(rf"(?<!\d){re.escape(canon)}(?!\d)", text):
            # Allow if it also appears in the original template path only via
            # shared operands — handled by caller comparing to original.
            return True
    return False


def validate_reverse_qa(
    *,
    original_prompt: str,
    rewritten_prompt: str,
    answer_spec: dict[str, Any],
    domain: str,
    min_arabic_ratio: float = 0.55,
    min_len: int = 12,
) -> tuple[bool, str]:
    """Gate a diversified prompt. Never mutates answer_spec."""
    new = (rewritten_prompt or "").strip()
    old = (original_prompt or "").strip()
    if not new:
        return False, "empty_rewrite"
    if new == old:
        return False, "unchanged"
    if len(new) < min_len:
        return False, "too_short"
    if arabic_char_ratio(new) < min_arabic_ratio:
        return False, "low_arabic"
    # Hard ban: answer string newly introduced (or present at all for short GT).
    for canon in _canonical_strings(answer_spec):
        if not canon:
            continue
        in_new = canon in new if len(canon) >= 6 else bool(
            re.search(rf"(?<!\d){re.escape(canon)}(?!\d)", new)
        )
        in_old = canon in old if len(canon) >= 6 else bool(
            re.search(rf"(?<!\d){re.escape(canon)}(?!\d)", old)
        )
        if in_new and not in_old:
            return False, "gt_leak"
        # Always reject very short numeric GT appearing as the sole focus leak
        # when the rewrite explicitly says the answer.
        if in_new and re.search(
            rf"(الإجابة| الناتج|الجواب|correct answer)\s*[:=]?\s*{re.escape(canon)}",
            new,
            flags=re.I,
        ):
            return False, "gt_leak_phrase"
    # Keep at least one shared digit token when original had digits (math fidelity).
    old_digits = set(_DIGIT_RE.findall(old))
    new_digits = set(_DIGIT_RE.findall(new))
    if old_digits and domain in {"gsm8k", "math", "math_comp"}:
        if not (old_digits & new_digits):
            return False, "lost_operands"
    return True, "ok"


def _extract_json_prompt(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    # Prefer fenced JSON
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
    # Plain text fallback
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_reverse_qa_messages(
    *,
    original_prompt: str,
    domain: str,
    answer_spec: dict[str, Any],
    latent: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
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


def make_openai_compatible_rewrite_fn(
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> RewriteFn:
    """Build a rewrite_fn using DeepSeek direct or OpenRouter (same routing as teacher)."""

    def _rewrite(
        original_prompt: str,
        domain: str,
        answer_spec: dict[str, Any],
        latent: dict[str, Any],
    ) -> str:
        from openai import OpenAI
        from synth.dspy_teacher import _resolve_lm_endpoint

        use_model = model or os.environ.get("REVERSE_QA_MODEL") or os.environ.get(
            "TEACHER_MODEL", "deepseek-v4-flash"
        )
        dspy_model, api_key, api_base = _resolve_lm_endpoint(use_model)
        # dspy model like openai/deepseek-v4-flash → API model id
        api_model = dspy_model.split("/", 1)[-1] if "/" in dspy_model else dspy_model
        client = OpenAI(api_key=api_key, base_url=api_base)
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


def diversify_rendered_problem(
    problem: RenderedProblem,
    *,
    rewrite_fn: RewriteFn,
    latent: dict[str, Any] | None = None,
    enabled: bool = True,
) -> RenderedProblem:
    """Return a diversified copy or the original on any failure."""
    if not enabled:
        return problem
    original = problem.prompt
    try:
        raw = rewrite_fn(
            original,
            problem.domain,
            dict(problem.answer_spec or {}),
            dict(latent or problem.metadata or {}),
        )
        raw = _extract_json_prompt(str(raw or ""))
    except Exception as exc:
        meta = dict(problem.metadata or {})
        meta["reverse_qa"] = {"applied": False, "reason": f"rewrite_error:{type(exc).__name__}"}
        return replace(
            problem,
            metadata=meta,
            provenance={**dict(problem.provenance or {}), "reverse_qa": False},
        )

    ok, reason = validate_reverse_qa(
        original_prompt=original,
        rewritten_prompt=raw,
        answer_spec=dict(problem.answer_spec or {}),
        domain=problem.domain,
    )
    meta = dict(problem.metadata or {})
    if not ok:
        meta["reverse_qa"] = {"applied": False, "reason": reason}
        return replace(
            problem,
            metadata=meta,
            provenance={**dict(problem.provenance or {}), "reverse_qa": False},
        )

    # Recompute problem_id from diversified prompt (family_id stays).
    from rlvr_synth.quality.ids import content_problem_id

    new_pid = content_problem_id(
        domain=problem.domain,
        prompt=raw.strip(),
        answer_canonical=(problem.answer_spec or {}).get("canonical"),
        partition=problem.partition,
        seed=int((problem.provenance or {}).get("seed") or 0),
    )
    meta["reverse_qa"] = {
        "applied": True,
        "reason": "ok",
        "original_prompt": original,
    }
    return replace(
        problem,
        problem_id=new_pid,
        prompt=raw.strip(),
        answer_spec=dict(problem.answer_spec or {}),  # frozen
        metadata=meta,
        provenance={
            **dict(problem.provenance or {}),
            "reverse_qa": True,
            "source": "programmatic+reverse_qa",
        },
    )


def reverse_qa_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    if "reverse_qa" in cfg:
        return bool(cfg.get("reverse_qa"))
    return os.environ.get("REVERSE_QA", "0").strip() in {"1", "true", "True", "yes"}
