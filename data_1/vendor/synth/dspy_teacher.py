"""DSPy multi-stage Formal Arabic teacher + GEPA-ready metric + budgeted DeepSeek LMs."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv
from dspy.utils.callback import BaseCallback

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from vendor import answer_match as V
from vendor.formal_saudi_style import formal_saudi_score, unique_ratio_think
from synth.dspy_signatures import (
    CompactThink,
    CompleteTruncatedSteps,
    CritiqueTeacherDraft,
    MaterializeExplicitArithmetic,
    PlanSolutionSteps,
    PolishFinishEquations,
    RefineToGroundTruth,
    SolveTeacherCoT,
    StripFinalLeak,
)
from synth.programmatic import canonicalize_answer, scrub_think_text, think_leaks_final_gt

MODEL_PRO = "deepseek-v4-pro"
MODEL_FLASH = "deepseek-v4-flash"
BASE = "https://api.deepseek.com"
EXTRA = {"thinking": {"type": "disabled"}}

PRO_IN, PRO_OUT = 0.435 / 1_000_000, 0.87 / 1_000_000
FLASH_IN, FLASH_OUT = 0.14 / 1_000_000, 0.28 / 1_000_000

BANNED_PHRASES = (
    "الخطوة التوضيحية",
    "يمكن حل هذه المسألة باتباع الخطوات",
    "نراجع الاتساق قبل إخراج الجواب",
    "نراجع خطوات الحل ونتأكد من الاتساق",
    "نراجع الاتساق قبل إخراج الجواب في وسم الإجابة",
)

_INCOMPLETE_THINK_RE = re.compile(
    r"(?:^|\n)\s*(?:ثم\s+)?(?:نضرب|نجمع|نطرح|نقسم|فيكون)\s*$",
    re.MULTILINE,
)
_DANGLING_CONNECTOR_RE = re.compile(
    r"(?:فيكون|فيصبح|ويكون|فيساوي)\s*\.(?!\d)|(?:فيكون|فيصبح)\s*$",
    re.MULTILINE,
)
# Bare binary op not closed with = / يساوي before clause end (., ،, newline, or EOS).
# Note: '/' omitted — Arabic fractions like 1/10 and 25/100 are not bare ops.
_NUM = r"\d+(?:\.\d+)?"
_BARE_OP_RE = re.compile(
    rf"(?<!\d){_NUM}\s*"
    r"(?:[×xX*+\-−÷]|زائد|ناقص|مضروب(?:ًا|اً|ة)?\s+في|مقسوم(?:ًا|اً|ة)?\s+على)"
    rf"\s*{_NUM}"
    rf"(?!\s*(?:[×xX*+\-−÷]|زائد|ناقص)\s*{_NUM})"  # not mid multi-factor chain
    rf"(?!\s*(?:=|يساوي)\s*-?{_NUM})"
    r"(?=\s*[.،]|\s*$|\s*\n)"
)
_EASTERN_DIGITS_RE = re.compile(r"[٠-٩۰-۹]")


def arabic_purity(text: str) -> float:
    if not text:
        return 0.0
    letters = re.findall(r"[\u0600-\u06ffA-Za-z]", text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0600" <= c <= "\u06ff") / len(letters)


def assemble_response(think: str, answer: str) -> str:
    return f"<think>\n{(think or '').strip()}\n</think>\n<answer>{(answer or '').strip()}</answer>"


def extract_think_answer(resp: str, ans_field: Any = None) -> tuple[str, Any]:
    think = ""
    m = re.search(r"<think>(.*?)</think>", resp or "", re.DOTALL | re.IGNORECASE)
    if m:
        think = m.group(1).strip()
    final = ans_field
    am = re.search(r"<answer>(.*?)</answer>", resp or "", re.DOTALL | re.IGNORECASE)
    if am:
        parsed = am.group(1).strip()
        if final is None:
            final = parsed
        if isinstance(final, str) and final.strip().startswith("{"):
            try:
                final = json.loads(final)
            except json.JSONDecodeError:
                pass
        elif final is None:
            final = parsed
    return think, final


def match_gt(domain: str, model_ans: Any, gt: Any) -> bool:
    if domain in ("gsm8k", "math"):
        return V.answers_match_numeric(model_ans, gt)
    if domain == "math_comp":
        st = V.answers_match_math_comp(model_ans, gt)
        if st == "match":
            return True
        return V.answers_match_numeric(model_ans, gt)
    if domain == "logic":
        if V.answers_match_logic(model_ans, gt):
            return True
        if isinstance(gt, dict) and model_ans is not None:
            text = (
                json.dumps(model_ans, ensure_ascii=False)
                if not isinstance(model_ans, str)
                else model_ans
            )
            return all(
                str(v) in text and (str(k) in text or str(k).split("_")[0] in text)
                for k, v in gt.items()
            )
        return False
    return False


def think_is_incomplete(think: str) -> bool:
    if not think or not think.strip():
        return True
    if _INCOMPLETE_THINK_RE.search(think.strip()):
        return True
    if _DANGLING_CONNECTOR_RE.search(think):
        return True
    last = [ln.strip() for ln in think.splitlines() if ln.strip()]
    if last and re.fullmatch(r"(?:ثم\s+)?(?:نضرب|نجمع|نطرح|نقسم|فيكون)\.?", last[-1]):
        return True
    # Hollow "فيكون ." mid-prose (space/dot with no following digit)
    if re.search(r"فيكون\s*\.\s*(?!\d)", think):
        return True
    return False


# Explicit a op b = c (or يساوي). Catches distillation-ready intermediates.
_EXPLICIT_EQ_RE = re.compile(
    r"\d+\s*(?:[×xX*+\-−÷/]|مضروب(?:ًا|اً|ة)?\s+في|زائد|ناقص|مقسوم(?:ًا|اً|ة)?\s+على)"
    r"\s*\d+\s*(?:=|يساوي)\s*-?\d+"
)
_VAGUE_RESULT_RE = re.compile(
    r"(?:نحصل على (?:الناتج|المبلغ|القيمة|المجموع)|بعد إيجاد ناتج|"
    r"لنحصل على الناتج(?: النهائي)?|نصل إلى (?:القيمة|الناتج) الصحيحة|"
    r"فيصبح الناتج|هذا هو الناتج)"
)
_LOGIC_ASSIGN_RE = re.compile(
    r"(?:المركز|المرتبة)\s*(?:الأول|الثاني|الثالث|الرابع)|"
    r"(?:هو|هي)\s+(?:الصادق|الكاذب|المتقلب|الأول|الأخير)"
)
_ANSWER_AS_RHS_RE_TMPL = r"(?:=|يساوي)\s*{ans}\b"


def _numeric_answer_token(gt: Any) -> str | None:
    """Return canonical integer/decimal token for numeric GT, else None."""
    if isinstance(gt, bool):
        return None
    if isinstance(gt, int):
        return str(gt)
    if isinstance(gt, float):
        if gt.is_integer():
            return str(int(gt))
        return str(gt).rstrip("0").rstrip(".") if "." in str(gt) else str(gt)
    if isinstance(gt, str):
        s = gt.strip()
        if re.fullmatch(r"-?\d+", s):
            return s
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", s)
        if m and not s.strip().startswith("{"):
            tok = m.group(1)
            if "." in tok:
                tok = tok.rstrip("0").rstrip(".")
            return tok
    return None


def think_has_explicit_intermediates(think: str, domain: str = "") -> bool:
    """True iff think shows concrete intermediates (not vague narration)."""
    if not think or not think.strip():
        return False
    if domain == "logic":
        return bool(_LOGIC_ASSIGN_RE.search(think)) and len(_LOGIC_ASSIGN_RE.findall(think)) >= 2
    return len(_EXPLICIT_EQ_RE.findall(think)) >= 1


def think_is_vague_narration(think: str, domain: str = "") -> bool:
    """Vague if it talks about results without writing enough explicit equations."""
    if not think:
        return True
    if domain == "logic":
        return not think_has_explicit_intermediates(think, domain)
    eqs = len(_EXPLICIT_EQ_RE.findall(think))
    vague = len(_VAGUE_RESULT_RE.findall(think))
    if eqs == 0:
        return True
    if vague >= 2 and eqs < 2:
        return True
    return False


def think_has_bare_ops(think: str) -> bool:
    """True if a clause-final a op b is missing '= c'."""
    if not think:
        return False
    return bool(_BARE_OP_RE.search(think))


def think_has_answer_equation(think: str, gt: Any, domain: str = "") -> bool:
    """Numeric: final answer must appear as an equation RHS (= N / يساوي N)."""
    if domain == "logic":
        return True
    ans = _numeric_answer_token(gt)
    if not ans:
        return True
    return bool(re.search(_ANSWER_AS_RHS_RE_TMPL.format(ans=re.escape(ans)), think))


def think_needs_polish(think: str, gt: Any = None, domain: str = "") -> str | None:
    """Return polish failure tag, or None if think is polish-clean."""
    if not think:
        return "empty"
    if think_is_incomplete(think):
        return "dangling_connector"
    if domain != "logic" and think_has_bare_ops(think):
        return "bare_op"
    if gt is not None and not think_has_answer_equation(think, gt, domain):
        return "missing_answer_eq"
    return None


def think_quality_ok(
    think: str,
    response: str,
    domain: str = "gsm8k",
    gt: Any = None,
) -> tuple[bool, str]:
    tokens = think.split()
    if len(tokens) < 20:
        return False, "think_too_short"
    if unique_ratio_think(response) < 0.55:
        return False, "unique_ratio"
    if arabic_purity(think) < 0.85:
        return False, "arabic_purity"
    for p in BANNED_PHRASES:
        if p in think:
            return False, f"boilerplate:{p}"
    if think_is_incomplete(think):
        return False, "incomplete"
    if _EASTERN_DIGITS_RE.search(think):
        return False, "eastern_digits"
    if "####" in response:
        return False, "hash_marker"
    if think_is_vague_narration(think, domain):
        return False, "vague_narration"
    # Soft polish gaps (bare_op / missing_answer_eq) do NOT fail ship quality —
    # polish stage best-effort repairs them; materialize-grade CoT is still usable.
    score, _ = formal_saudi_score(think, domain="gsm8k")
    if score < 0.45:
        return False, "formal_score"
    return True, "ok"


def gt_str(gt: Any) -> str:
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False, sort_keys=True)
    return str(gt)


def parse_gt(gt: Any) -> Any:
    if isinstance(gt, str) and gt.strip().startswith("{"):
        try:
            return json.loads(gt)
        except json.JSONDecodeError:
            return gt
    return gt


class BudgetState:
    """Thread-safe USD spend tracker for DSPy + Flash calls."""

    def __init__(self, path: Path, budget_usd: float):
        self.path = path
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            st = json.loads(self.path.read_text(encoding="utf-8-sig"))
        else:
            st = {
                "spent_usd": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "calls": 0,
                "pro_calls": 0,
                "flash_calls": 0,
                "budget_usd": self.budget_usd,
                "cap_hit": False,
            }
        st["budget_usd"] = self.budget_usd
        return st

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["budget_usd"] = self.budget_usd
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def ok(self, soft_reserve: float = 0.30) -> bool:
        return float(self.data.get("spent_usd", 0.0)) < self.budget_usd - soft_reserve

    def record(self, *, model: str, tin: int, tout: int) -> None:
        with self._lock:
            if model == MODEL_PRO or "deepseek-v4-pro" in model:
                self.data["spent_usd"] = float(self.data.get("spent_usd", 0.0)) + tin * PRO_IN + tout * PRO_OUT
                self.data["pro_calls"] = int(self.data.get("pro_calls", 0)) + 1
            else:
                self.data["spent_usd"] = float(self.data.get("spent_usd", 0.0)) + tin * FLASH_IN + tout * FLASH_OUT
                self.data["flash_calls"] = int(self.data.get("flash_calls", 0)) + 1
            self.data["tokens_in"] = int(self.data.get("tokens_in", 0)) + tin
            self.data["tokens_out"] = int(self.data.get("tokens_out", 0)) + tout
            self.data["calls"] = int(self.data.get("calls", 0)) + 1
            self.save()


class BudgetCallback(BaseCallback):
    """Charge DeepSeek usage from DSPy LM history after each LM call."""

    def __init__(self, budget: BudgetState, model_hint: str = MODEL_PRO):
        self.budget = budget
        self.model_hint = model_hint
        self._seen: set[int] = set()

    def on_lm_end(self, call_id: str, outputs: Any, exception: Exception | None = None) -> None:
        if exception is not None:
            return
        lm = dspy.settings.lm
        if lm is None or not getattr(lm, "history", None):
            return
        entry = lm.history[-1]
        eid = id(entry)
        if eid in self._seen:
            return
        self._seen.add(eid)
        usage = entry.get("usage") or {}
        tin = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tout = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        model = str(entry.get("model") or getattr(lm, "model", None) or self.model_hint)
        if tin or tout:
            if not self.budget.ok():
                raise RuntimeError("BUDGET_CAP_HIT")
            self.budget.record(model=model, tin=tin, tout=tout)


def make_deepseek_lm(
    *,
    model: str = MODEL_PRO,
    temperature: float = 0.35,
    max_tokens: int = 3200,
    budget: BudgetState | None = None,
    cache: bool = True,
) -> dspy.LM:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    callbacks = [BudgetCallback(budget, model_hint=model)] if budget is not None else None
    return dspy.LM(
        model=f"openai/{model}",
        api_key=key,
        api_base=BASE,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        callbacks=callbacks,
        extra_body=EXTRA,
    )


def configure_teacher_lm(
    *,
    budget: BudgetState | None = None,
    temperature: float = 0.35,
    cache: bool = True,
    max_tokens: int = 3200,
) -> dspy.LM:
    """Configure global DSPy LM. ChatAdapter works reliably with DeepSeek V4."""
    lm = make_deepseek_lm(
        model=MODEL_PRO,
        temperature=temperature,
        max_tokens=max_tokens,
        budget=budget,
        cache=cache,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    return lm


_PREDICT_FAIL_N = 0


def _safe_predict(predictor, **kwargs):
    global _PREDICT_FAIL_N
    try:
        return predictor(**kwargs)
    except Exception as e:
        _PREDICT_FAIL_N += 1
        if _PREDICT_FAIL_N <= 5 or _PREDICT_FAIL_N % 25 == 0:
            print(
                f"[dspy_teacher] predict_fail#{_PREDICT_FAIL_N} {type(e).__name__}: {str(e)[:120]}",
                flush=True,
            )
        return None


def _from_split(pred) -> tuple[str, str]:
    """Pull think/answer from split-field prediction (or legacy response)."""
    if pred is None:
        return "", ""
    think = (getattr(pred, "think", None) or "").strip()
    answer = getattr(pred, "answer", None)
    if answer is None:
        answer = ""
    elif not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    if not think:
        resp = getattr(pred, "response", None) or ""
        t2, a2 = extract_think_answer(resp, answer or None)
        if t2:
            think = t2
        if a2 is not None and not answer:
            answer = a2 if isinstance(a2, str) else json.dumps(a2, ensure_ascii=False)
    return think, answer


class ArabicTeacher(dspy.Module):
    """plan → solve → refine → strip → complete → materialize → polish → compact → assemble."""

    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanSolutionSteps)
        self.solve = dspy.Predict(SolveTeacherCoT)
        self.refine = dspy.Predict(RefineToGroundTruth)
        self.strip_leak = dspy.Predict(StripFinalLeak)
        self.complete = dspy.Predict(CompleteTruncatedSteps)
        self.materialize = dspy.Predict(MaterializeExplicitArithmetic)
        self.polish = dspy.Predict(PolishFinishEquations)
        self.compact = dspy.Predict(CompactThink)
        self.critic = dspy.Predict(CritiqueTeacherDraft)

    def forward(self, domain: str, problem: str, ground_truth: str = ""):
        gt_obj = parse_gt(ground_truth)
        gts = gt_str(gt_obj) if ground_truth != "" else ""

        plan_out = _safe_predict(self.plan, domain=domain, problem=problem)
        plan_text = ((plan_out.plan if plan_out else "") or "").strip()

        think, answer = "", ""
        for _ in range(2):
            solved = _safe_predict(self.solve, domain=domain, problem=problem, plan=plan_text)
            think, answer = _from_split(solved)
            if think:
                break

        final: Any = answer
        if isinstance(answer, str) and answer.startswith("{"):
            try:
                final = json.loads(answer)
            except json.JSONDecodeError:
                final = answer

        if gts and (not think or not match_gt(domain, final, gt_obj)):
            for _ in range(2):
                refined = _safe_predict(
                    self.refine,
                    domain=domain,
                    problem=problem,
                    ground_truth=gts,
                    draft_think=think,
                    draft_answer=answer or str(final or ""),
                )
                if refined is None:
                    # legacy field names fallback
                    refined = _safe_predict(
                        self.refine,
                        domain=domain,
                        problem=problem,
                        ground_truth=gts,
                        draft_response=assemble_response(think, answer),
                    )
                if refined is None:
                    break
                think, answer = _from_split(refined)
                final = answer
                if isinstance(answer, str) and answer.startswith("{"):
                    try:
                        final = json.loads(answer)
                    except json.JSONDecodeError:
                        pass
                if think and match_gt(domain, final, gt_obj):
                    break

        if gts and think and match_gt(domain, final, gt_obj) and think_leaks_final_gt(think, gt_obj, domain):
            trimmed = scrub_think_text(think, gt_obj, domain)
            if trimmed and not think_leaks_final_gt(trimmed, gt_obj, domain) and not think_is_incomplete(trimmed):
                think = trimmed
                answer = canonicalize_answer(gt_obj, domain)
                final = gt_obj
            else:
                stripped = _safe_predict(
                    self.strip_leak,
                    domain=domain,
                    problem=problem,
                    ground_truth=gts,
                    draft_think=think,
                    draft_answer=answer,
                )
                if stripped is not None:
                    think, answer = _from_split(stripped)
                    final = answer
                if think and think_leaks_final_gt(think, gt_obj, domain):
                    trimmed2 = scrub_think_text(think, gt_obj, domain)
                    if trimmed2 and not think_leaks_final_gt(trimmed2, gt_obj, domain):
                        think = trimmed2
                        answer = canonicalize_answer(gt_obj, domain)
                        final = gt_obj
                    else:
                        # empty/leaky after scrub → reject by clearing (metric fails)
                        think = ""

        if think and think_is_incomplete(think) and gts:
            completed = _safe_predict(
                self.complete,
                domain=domain,
                problem=problem,
                ground_truth=gts,
                draft_think=think,
            )
            if completed is not None:
                c_think, c_ans = _from_split(completed)
                if c_think and not think_is_incomplete(c_think):
                    if not think_leaks_final_gt(c_think, gt_obj, domain):
                        think = c_think
                        if c_ans:
                            answer = c_ans
                            final = c_ans

        # Fix distillation-killing vague narration: force a op b = c lines
        if think and gts and think_is_vague_narration(think, domain):
            materialized = _safe_predict(
                self.materialize,
                domain=domain,
                problem=problem,
                ground_truth=gts,
                draft_think=think,
                draft_answer=answer or canonicalize_answer(gt_obj, domain),
            )
            if materialized is not None:
                m_think, m_ans = _from_split(materialized)
                if (
                    m_think
                    and not think_is_incomplete(m_think)
                    and not think_is_vague_narration(m_think, domain)
                    and match_gt(domain, m_ans or final, gt_obj)
                    and not think_leaks_final_gt(m_think, gt_obj, domain)
                ):
                    think = m_think
                    answer = m_ans or canonicalize_answer(gt_obj, domain)
                    final = answer

        # Polish soft gaps (best-effort): retry once; keep materialize draft if polish fails.
        if think and gts and think_needs_polish(think, gt=gt_obj, domain=domain):
            draft_think, draft_answer = think, answer or canonicalize_answer(gt_obj, domain)
            for _ in range(2):
                polished = _safe_predict(
                    self.polish,
                    domain=domain,
                    problem=problem,
                    ground_truth=gts,
                    draft_think=draft_think,
                    draft_answer=draft_answer,
                )
                if polished is None:
                    break
                p_think, p_ans = _from_split(polished)
                if not p_think:
                    break
                if think_is_incomplete(p_think) or think_is_vague_narration(p_think, domain):
                    # Bad polish — ignore, keep prior draft
                    break
                if not match_gt(domain, p_ans or final, gt_obj):
                    break
                if think_leaks_final_gt(p_think, gt_obj, domain):
                    break
                # Accept improved draft even if still soft-gap; prefer fully clean.
                think = p_think
                answer = p_ans or canonicalize_answer(gt_obj, domain)
                final = answer
                draft_think, draft_answer = think, answer
                if not think_needs_polish(think, gt=gt_obj, domain=domain):
                    break

        response = assemble_response(think, answer) if think else ""
        if think and response and unique_ratio_think(response) < 0.55 and gts:
            compacted = _safe_predict(
                self.compact,
                domain=domain,
                problem=problem,
                ground_truth=gts,
                draft_think=think,
            )
            if compacted is not None:
                c_think, c_ans = _from_split(compacted)
                if c_think and match_gt(domain, c_ans or final, gt_obj):
                    if (
                        not think_leaks_final_gt(c_think, gt_obj, domain)
                        and not think_is_incomplete(c_think)
                        and not think_is_vague_narration(c_think, domain)
                    ):
                        # Keep compact even with soft polish gaps
                        think = c_think
                        answer = c_ans or canonicalize_answer(gt_obj, domain)
                        response = assemble_response(think, answer)

        # Hard reject only: empty / incomplete / vague. Soft polish gaps may ship.
        if not think or think_is_incomplete(think) or think_is_vague_narration(think, domain):
            think = ""
            response = ""
            answer = ""
        else:
            if not isinstance(answer, str) or not answer:
                answer = canonicalize_answer(final if final is not None else gt_obj, domain)
            response = assemble_response(think, answer)

        return dspy.Prediction(
            response=response,
            answer=answer,
            plan=plan_text,
            think=think,
        )


def teacher_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """GEPA metric: score 1.0 only if parse OK + match GT + no leak + think quality."""
    domain = getattr(gold, "domain", "") or ""
    gt_raw = getattr(gold, "ground_truth", None)
    if gt_raw is None:
        gt_raw = getattr(gold, "answer", None)
    gt = parse_gt(gt_raw)

    response = getattr(pred, "response", "") or ""
    ans_field = getattr(pred, "answer", None)
    think = (getattr(pred, "think", None) or "").strip()
    if not think:
        think, final = extract_think_answer(response, ans_field)
    else:
        final = ans_field
        if isinstance(final, str) and final.startswith("{"):
            try:
                final = json.loads(final)
            except json.JSONDecodeError:
                pass
        if not response:
            response = assemble_response(think, str(ans_field or ""))

    feedback_bits: list[str] = []
    score = 0.0

    if not think or "<think>" not in response.lower() or "<answer>" not in response.lower():
        feedback_bits.append("parse_fail: missing <think>/<answer> or empty think")
    elif "####" in response:
        feedback_bits.append("hash_marker")
    elif not match_gt(domain, final, gt):
        feedback_bits.append(f"answer!=gt: got {final!r} want {gt!r}")
    elif think_leaks_final_gt(think, gt, domain):
        last = think.strip().splitlines()[-1] if think.strip() else ""
        feedback_bits.append(f"leak: final GT in think (last~{last[:100]!r})")
    else:
        ok, reason = think_quality_ok(think, response, domain=domain, gt=gt)
        if not ok:
            feedback_bits.append(reason)
        else:
            score = 1.0
            feedback_bits.append("ok: match GT, no leak, think quality pass")

    if pred_name and score < 1.0:
        feedback_bits.append(
            f"component={pred_name}: Formal Arabic CoT; Western digits; "
            "explicit a op b = c; answer as equation RHS; no bare ops; "
            "no vague 'الناتج'; no إذن/لذلك closer; no الاتساق filler"
        )

    return dspy.Prediction(score=score, feedback="; ".join(feedback_bits))


def score_prediction(domain: str, response: str, answer: Any, gt: Any) -> tuple[float, str]:
    ex = dspy.Example(domain=domain, ground_truth=gt_str(gt), answer=gt_str(gt))
    pred = dspy.Prediction(response=response, answer=answer)
    out = teacher_metric(ex, pred)
    return float(out.score), str(out.feedback)


def load_teacher(path: Path) -> ArabicTeacher:
    if not path.exists():
        raise FileNotFoundError(f"compiled teacher missing: {path}")
    teacher = ArabicTeacher()
    try:
        teacher.load(str(path))
    except KeyError as e:
        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0
        for name, pred in teacher.named_predictors():
            if name in raw:
                pred.load_state(raw[name])
                loaded += 1
        if loaded == 0:
            raise RuntimeError(f"could not load any predictors from {path}") from e
        print(f"[dspy_teacher] partial load from {path.name}: {loaded} predictors ({e})", flush=True)
    return teacher
