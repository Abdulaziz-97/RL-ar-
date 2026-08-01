"""Tests for constraint_set / IFEval-like contracts (no MCQ generation)."""

from __future__ import annotations

import random
import unittest

from rlvr_contracts.answer_spec import parse_answer_spec
from rlvr_contracts.constraints import check_constraint
from rlvr_contracts.verifiers import verify_answer
from vendor.synth.programmatic import gen_ifeval_multiconstraint


class TestConstraintContracts(unittest.TestCase):
    def test_word_and_bullet_checks(self):
        text = "واحد اثنان ثلاثة\n- بند أ\n- بند ب\n- بند ج\nملاحظة: تم"
        self.assertTrue(check_constraint("word_count_range", text, {"min": 3, "max": 20}))
        self.assertTrue(check_constraint("exact_bullets", text, {"count": 3}))
        self.assertTrue(check_constraint("endswith_line_prefix", text, {"prefix": "ملاحظة:"}))
        self.assertTrue(check_constraint("forbidden_substring", text, {"text": "حاسوب"}))

    def test_generator_teacher_verifies(self):
        rng = random.Random(123)
        failures = 0
        for _ in range(20):
            sample = gen_ifeval_multiconstraint(rng)
            spec = sample.meta_extra["answer_spec"]
            body = sample.meta_extra["teacher_body"]
            parsed = parse_answer_spec(spec)
            self.assertEqual(parsed.type, "constraint_set")
            result = verify_answer(body, spec)
            if not result.ok:
                failures += 1
        self.assertLessEqual(failures, 2, "too many unverifiable teacher bodies")

    def test_wrapped_response_verifies(self):
        rng = random.Random(7)
        sample = gen_ifeval_multiconstraint(rng)
        body = sample.meta_extra["teacher_body"]
        completion = f"<think>خطوات</think><answer>\n{body}\n</answer>"
        result = verify_answer(
            completion, sample.meta_extra["answer_spec"], from_completion=True
        )
        self.assertTrue(result.ok, result.reason)


if __name__ == "__main__":
    unittest.main()
