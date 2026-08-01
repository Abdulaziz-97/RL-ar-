"""Tests for Reverse-QA prompt diversification (solver GT fixed, LLM surface only)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor")]

from rlvr_synth.roles.protocols import RenderedProblem
from rlvr_synth.roles.reverse_qa import (
    build_reverse_qa_messages,
    diversify_rendered_problem,
    reverse_qa_enabled,
    validate_reverse_qa,
)


def _spec(canonical="42", typ="integer") -> dict:
    return {"type": typ, "canonical": canonical, "ground_truth_structured": int(canonical) if typ == "integer" else canonical}


class ValidateReverseQATests(unittest.TestCase):
    def test_accepts_diverse_arabic_keeping_operands(self) -> None:
        old = "اشترى تاجر 12 قلمًا بسعر 3 ريالات للقلم. كم دفع إجمالًا؟"
        new = "تاجر اشترى اثني عشر قلمًا، ثمن القلم الواحد 3 ريالات. ما مجموع المبلغ؟"
        # keep digits 12 and 3; answer 36 must not appear
        new = "اشترى أحمد 12 قلمًا بسعر 3 ريالات للواحد. ما إجمالي الثمن؟"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=new,
            answer_spec=_spec("36"),
            domain="gsm8k",
        )
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "ok")

    def test_rejects_gt_leak_introduced(self) -> None:
        old = "ما مجموع 10 و 5؟"
        new = "احسب 10 زائد 5 والجواب 15"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=new,
            answer_spec=_spec("15"),
            domain="gsm8k",
        )
        self.assertFalse(ok)
        self.assertIn("gt_leak", reason)

    def test_rejects_empty_and_unchanged(self) -> None:
        old = "مسألة عربية طويلة بما يكفي للاختبار رقم 7 و 8"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt="   ",
            answer_spec=_spec("3"),
            domain="math",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "empty_rewrite")
        ok2, reason2 = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=old,
            answer_spec=_spec("3"),
            domain="math",
        )
        self.assertFalse(ok2)
        self.assertEqual(reason2, "unchanged")

    def test_rejects_lost_math_operands(self) -> None:
        old = "ما ناتج 18 × 4؟"
        new = "ما ناتج ضرب عددين مختلفين تمامًا عن الأصل؟"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=new,
            answer_spec=_spec("72"),
            domain="math",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "lost_operands")

    def test_rejects_low_arabic(self) -> None:
        old = "ما مجموع العددین 2 و 3؟"
        new = "What is 2 plus 3 in this homework problem please?"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=new,
            answer_spec=_spec("5"),
            domain="gsm8k",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_arabic")


class DiversifyRenderedProblemTests(unittest.TestCase):
    def _problem(self, prompt: str, canonical: str = "36") -> RenderedProblem:
        return RenderedProblem(
            problem_id="p0",
            family_id="fam0",
            domain="gsm8k",
            partition="sft_train",
            prompt=prompt,
            answer_spec=_spec(canonical),
            provenance={"source": "programmatic", "seed": 7},
            metadata={"ground_truth": int(canonical), "solution_steps": ["خطوة"]},
        )

    def test_applies_rewrite_keeps_answer_spec(self) -> None:
        old = "اشترى تاجر 12 قلمًا بسعر 3 ريالات للقلم. كم دفع إجمالًا؟"
        new = "اشترى خالد 12 قلمًا، سعر القلم 3 ريالات. ما المبلغ الكلي؟"

        def rewrite(prompt, domain, spec, latent):
            self.assertEqual(prompt, old)
            self.assertEqual(spec["canonical"], "36")
            return json.dumps({"prompt": new}, ensure_ascii=False)

        out = diversify_rendered_problem(
            self._problem(old),
            rewrite_fn=rewrite,
            latent={"solution_steps": ["12*3"]},
            enabled=True,
        )
        self.assertEqual(out.prompt, new)
        self.assertEqual(out.answer_spec["canonical"], "36")
        self.assertEqual(out.family_id, "fam0")
        self.assertNotEqual(out.problem_id, "p0")
        self.assertTrue(out.provenance.get("reverse_qa"))
        self.assertTrue(out.metadata["reverse_qa"]["applied"])

    def test_fallback_on_bad_rewrite(self) -> None:
        old = "اشترى تاجر 12 قلمًا بسعر 3 ريالات للقلم. كم دفع إجمالًا؟"

        def rewrite(prompt, domain, spec, latent):
            return "الجواب النهائي هو 36 فقط"

        out = diversify_rendered_problem(
            self._problem(old),
            rewrite_fn=rewrite,
            enabled=True,
        )
        self.assertEqual(out.prompt, old)
        self.assertEqual(out.answer_spec["canonical"], "36")
        self.assertFalse(out.metadata["reverse_qa"]["applied"])

    def test_fallback_on_rewrite_exception(self) -> None:
        old = "مسألة عربية كافية الطول فيها العددان 9 و 4"

        def rewrite(prompt, domain, spec, latent):
            raise RuntimeError("api down")

        out = diversify_rendered_problem(
            self._problem(old, canonical="13"),
            rewrite_fn=rewrite,
            enabled=True,
        )
        self.assertEqual(out.prompt, old)
        self.assertIn("rewrite_error", out.metadata["reverse_qa"]["reason"])

    def test_disabled_noop(self) -> None:
        old = "نص عربي اختباري مع رقم 2"

        def rewrite(*args, **kwargs):
            raise AssertionError("should not be called")

        out = diversify_rendered_problem(
            self._problem(old, canonical="99"),
            rewrite_fn=rewrite,
            enabled=False,
        )
        self.assertEqual(out.prompt, old)


class LiveGeneratorReverseQATests(unittest.TestCase):
    def test_live_generator_uses_injected_rewrite(self) -> None:
        from backends.dspy_backend import LiveProblemGenerator

        def rewrite(prompt, domain, spec, latent):
            base = prompt.rstrip("؟").rstrip()
            return f"{base} — أعد صياغة السؤال بصيغة مختلفة قليلًا؟"

        gen = LiveProblemGenerator(
            {
                "reverse_qa": True,
                "reverse_qa_rewrite_fn": rewrite,
            }
        )
        latent = gen.generate_family("gsm8k", seed=1200)
        rendered = gen.render_arabic(latent, seed=1200)
        self.assertEqual(rendered.answer_spec, latent.answer_spec)
        self.assertEqual(rendered.family_id, latent.family_id)
        self.assertEqual(
            rendered.answer_spec.get("canonical"),
            latent.answer_spec.get("canonical"),
        )
        self.assertIn("reverse_qa", rendered.metadata)
        self.assertTrue(rendered.metadata["reverse_qa"]["applied"])
        self.assertNotEqual(rendered.prompt, latent.latent["prompt"])

    def test_reverse_qa_disabled_by_config(self) -> None:
        from backends.dspy_backend import LiveProblemGenerator

        gen = LiveProblemGenerator({"reverse_qa": False})
        self.assertFalse(gen._reverse_qa)
        latent = gen.generate_family("math", seed=5)
        rendered = gen.render_arabic(latent, seed=5)
        self.assertEqual(rendered.prompt, latent.latent["prompt"])
        self.assertNotIn("reverse_qa", rendered.metadata)


class HelpersTests(unittest.TestCase):
    def test_messages_include_fixed_answer_and_instructions(self) -> None:
        msgs = build_reverse_qa_messages(
            original_prompt="سؤال",
            domain="logic",
            answer_spec=_spec('{"a":1}', typ="logic_json"),
            latent={"solution_steps": ["س1"]},
        )
        self.assertEqual(msgs[0]["role"], "system")
        body = json.loads(msgs[1]["content"])
        self.assertEqual(body["domain"], "logic")
        self.assertIn("Do NOT include the final answer", body["instructions"][2])

    def test_env_flag_parsing(self) -> None:
        self.assertTrue(reverse_qa_enabled({"reverse_qa": True}))
        self.assertFalse(reverse_qa_enabled({"reverse_qa": False}))
        with patch.dict("os.environ", {"REVERSE_QA": "1"}, clear=False):
            self.assertTrue(reverse_qa_enabled({}))
        with patch.dict("os.environ", {"REVERSE_QA": "0"}, clear=False):
            self.assertFalse(reverse_qa_enabled({}))


class OrchestratorRenderGateTests(unittest.TestCase):
    """Diversified prompt still passes independent problem verifier; GT frozen."""

    def test_verifier_accepts_diversified_keeps_spec(self) -> None:
        from rlvr_synth.roles.stub import StubIndependentVerifier

        old = "ما مجموع العددین 8 و 7 في المسألة التالية؟"
        new = "احسب حاصل جمع العددين 8 و 7 ضمن هذه المسألة العربية."
        problem = RenderedProblem(
            problem_id="p1",
            family_id="f1",
            domain="math",
            partition="rlvr_train",
            prompt=old,
            answer_spec=_spec("15"),
            provenance={"seed": 1},
            metadata={},
        )

        def rewrite(prompt, domain, spec, latent):
            return json.dumps({"prompt": new}, ensure_ascii=False)

        out = diversify_rendered_problem(problem, rewrite_fn=rewrite, enabled=True)
        ok, reasons = StubIndependentVerifier().verify_problem(out)
        self.assertTrue(ok, reasons)
        self.assertEqual(out.answer_spec["canonical"], "15")
        self.assertEqual(out.prompt, new)


if __name__ == "__main__":
    unittest.main()
