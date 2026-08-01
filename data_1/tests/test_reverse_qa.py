"""Tests for full answer-first Reverse-QA + paraphrase fallback."""

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
    answers_match,
    build_full_reverse_qa_messages,
    build_reverse_qa_messages,
    diversify_rendered_problem,
    reverse_qa_enabled,
    reverse_qa_mode,
    validate_generated_question,
    validate_reverse_qa,
)


def _spec(canonical="42", typ="integer") -> dict:
    return {
        "type": typ,
        "canonical": canonical,
        "ground_truth_structured": int(canonical) if typ == "integer" and str(canonical).isdigit() else canonical,
    }


class ValidateReverseQATests(unittest.TestCase):
    def test_accepts_diverse_arabic_keeping_operands(self) -> None:
        old = "اشترى تاجر 12 قلمًا بسعر 3 ريالات للقلم. كم دفع إجمالًا؟"
        new = "اشترى أحمد 12 قلمًا بسعر 3 ريالات للواحد. ما إجمالي الثمن؟"
        ok, reason = validate_reverse_qa(
            original_prompt=old,
            rewritten_prompt=new,
            answer_spec=_spec("36"),
            domain="gsm8k",
        )
        self.assertTrue(ok, reason)

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


class FullModeValidateTests(unittest.TestCase):
    def test_full_accepts_new_question_with_operands(self) -> None:
        ok, reason = validate_generated_question(
            prompt="وزّع معلم 12 كتابًا على 3 طلاب بالتساوي. كم كتابًا لكل طالب؟",
            answer_spec=_spec("4"),
            domain="gsm8k",
            latent={
                "solution_steps": ["12 / 3 = 4"],
                "meta_extra": {"qty": 12, "groups": 3},
            },
            template_prompt=None,
        )
        self.assertTrue(ok, reason)

    def test_full_rejects_answer_in_question(self) -> None:
        ok, reason = validate_generated_question(
            prompt="إذا كان الناتج 4 فما هو توزيع 12 على 3؟",
            answer_spec=_spec("4"),
            domain="gsm8k",
            latent={"solution_steps": ["12/3=4"], "meta_extra": {"qty": 12}},
        )
        self.assertFalse(ok)
        self.assertIn("gt_leak", reason)


class DiversifyParaphraseTests(unittest.TestCase):
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
            return json.dumps({"prompt": new}, ensure_ascii=False)

        out = diversify_rendered_problem(
            self._problem(old),
            rewrite_fn=rewrite,
            enabled=True,
            mode="paraphrase",
            require_resolve=False,
        )
        self.assertEqual(out.prompt, new)
        self.assertEqual(out.answer_spec["canonical"], "36")
        self.assertTrue(out.metadata["reverse_qa"]["applied"])

    def test_fallback_on_bad_rewrite(self) -> None:
        old = "اشترى تاجر 12 قلمًا بسعر 3 ريالات للقلم. كم دفع إجمالًا؟"

        def rewrite(prompt, domain, spec, latent):
            return "الجواب النهائي هو 36 فقط"

        out = diversify_rendered_problem(
            self._problem(old),
            rewrite_fn=rewrite,
            mode="paraphrase",
            require_resolve=False,
            enabled=True,
        )
        self.assertEqual(out.prompt, old)
        self.assertFalse(out.metadata["reverse_qa"]["applied"])


class DiversifyFullModeTests(unittest.TestCase):
    def test_full_generate_and_resolve(self) -> None:
        problem = RenderedProblem(
            problem_id="p0",
            family_id="fam0",
            domain="gsm8k",
            partition="sft_train",
            prompt="قالب أصلي: 12 قلمًا بـ 3 ريالات. كم الإجمالي؟",
            answer_spec=_spec("36"),
            provenance={"seed": 1},
            metadata={"solution_steps": ["12*3=36"], "meta_extra": {"qty": 12, "price": 3}},
        )

        def generate(domain, spec, latent):
            self.assertNotIn("قالب أصلي", json.dumps(latent, ensure_ascii=False))
            return json.dumps(
                {"prompt": "اشترى رجل 12 قلمًا، ثمن القلم 3 ريالات. ما مجموع ما دفعه؟"},
                ensure_ascii=False,
            )

        def solve(prompt, domain):
            return "36"

        out = diversify_rendered_problem(
            problem,
            generate_fn=generate,
            solve_fn=solve,
            latent=dict(problem.metadata),
            mode="full",
            require_resolve=True,
            enabled=True,
        )
        self.assertTrue(out.metadata["reverse_qa"]["applied"])
        self.assertEqual(out.answer_spec["canonical"], "36")
        self.assertIn("12", out.prompt)
        self.assertNotEqual(out.prompt, problem.prompt)

    def test_full_resolve_mismatch_falls_back(self) -> None:
        problem = RenderedProblem(
            problem_id="p0",
            family_id="fam0",
            domain="gsm8k",
            partition="sft_train",
            prompt="قالب: 12 و 3",
            answer_spec=_spec("36"),
            provenance={"seed": 1},
            metadata={"solution_steps": ["12*3"], "meta_extra": {"qty": 12, "price": 3}},
        )

        def generate(domain, spec, latent):
            return "اشترى تاجر 12 قلمًا بسعر 3. كم دفع؟"

        def solve(prompt, domain):
            return "99"

        out = diversify_rendered_problem(
            problem,
            generate_fn=generate,
            solve_fn=solve,
            latent=dict(problem.metadata),
            mode="full",
            require_resolve=True,
            enabled=True,
        )
        self.assertEqual(out.prompt, problem.prompt)
        self.assertEqual(out.metadata["reverse_qa"]["reason"], "resolve_mismatch")

    def test_answers_match_numeric(self) -> None:
        self.assertTrue(answers_match("36", _spec("36")))
        self.assertTrue(answers_match("36.0", _spec("36")))
        self.assertFalse(answers_match("35", _spec("36")))


class LiveGeneratorReverseQATests(unittest.TestCase):
    def test_live_generator_full_mode_injected(self) -> None:
        from backends.dspy_backend import LiveProblemGenerator

        def generate(domain, spec, latent):
            # Keep latent operands if present in solution text; else append marker.
            steps = " ".join(str(x) for x in (latent.get("solution_steps") or []))
            digits = "".join(ch if ch.isdigit() or ch.isspace() else " " for ch in steps)
            return f"مسألة جديدة بالعربية حول القيم {digits.strip()} أوضح المطلوب؟"

        def solve(prompt, domain):
            # Always "succeed" by returning whatever canonical the caller used — look up from prompt not available;
            # tests patch via reading last generated family through closure.
            return solve.expected  # type: ignore[attr-defined]

        gen = LiveProblemGenerator(
            {
                "reverse_qa": True,
                "reverse_qa_mode": "full",
                "reverse_qa_resolve": True,
                "reverse_qa_generate_fn": generate,
                "reverse_qa_solve_fn": solve,
            }
        )
        latent = gen.generate_family("gsm8k", seed=1200)
        solve.expected = str(latent.answer_spec.get("canonical"))  # type: ignore[attr-defined]
        rendered = gen.render_arabic(latent, seed=1200)
        self.assertEqual(rendered.answer_spec.get("canonical"), latent.answer_spec.get("canonical"))
        self.assertIn("reverse_qa", rendered.metadata)

    def test_reverse_qa_disabled_by_config(self) -> None:
        from backends.dspy_backend import LiveProblemGenerator

        gen = LiveProblemGenerator({"reverse_qa": False})
        self.assertFalse(gen._reverse_qa)
        latent = gen.generate_family("math", seed=5)
        rendered = gen.render_arabic(latent, seed=5)
        self.assertEqual(rendered.prompt, latent.latent["prompt"])


class HelpersTests(unittest.TestCase):
    def test_full_messages_are_answer_first(self) -> None:
        msgs = build_full_reverse_qa_messages(
            domain="gsm8k",
            answer_spec=_spec("36"),
            latent={"solution_steps": ["12*3"], "template_family": "buy"},
        )
        body = json.loads(msgs[1]["content"])
        self.assertEqual(body["task"], "reverse_qa_full")
        self.assertNotIn("original_prompt", body)
        self.assertEqual(body["fixed_answer"]["canonical"], "36")

    def test_paraphrase_messages_keep_original(self) -> None:
        msgs = build_reverse_qa_messages(
            original_prompt="سؤال",
            domain="logic",
            answer_spec=_spec('{"a":1}', typ="logic_json"),
        )
        body = json.loads(msgs[1]["content"])
        self.assertEqual(body["original_prompt"], "سؤال")

    def test_env_flag_parsing(self) -> None:
        self.assertTrue(reverse_qa_enabled({"reverse_qa": True}))
        self.assertEqual(reverse_qa_mode({"reverse_qa_mode": "full"}), "full")
        with patch.dict("os.environ", {"REVERSE_QA_MODE": "paraphrase"}, clear=False):
            self.assertEqual(reverse_qa_mode({}), "paraphrase")


if __name__ == "__main__":
    unittest.main()
