"""Smoke tests for prompt-identity decontamination."""

from __future__ import annotations

import unittest

from rlvr_synth.decontam.pipeline import LSHIndex, MinHashSketch, decontaminate_records


class PromptIdentityDecontamTests(unittest.TestCase):
    def test_lsh_bands_match_num_perm(self) -> None:
        mh = MinHashSketch()
        lsh = LSHIndex(bands=8, rows=4)
        self.assertEqual(mh.num_perm, lsh.bands * lsh.rows)
        sk = mh.sketch("سؤال تجريبي عن النسبة المئوية")
        lsh.add("a", sk)
        self.assertIn("a", lsh.query(sk))

    def test_sft_ref_rejects_rlvr_same_prompt_empty_response(self) -> None:
        """SFT ref with long CoT must reject RLVR row with same prompt + empty response."""
        prompt = "احسب ناتج 17 × 23 مع توضيح الخطوات."
        long_cot = (
            "<think>\n"
            + ("نضرب الآحاد ثم العشرات. " * 80)
            + "\n</think>\n"
            + "الجواب النهائي: 391"
        )
        sft_ref_prompt = prompt  # orchestrator now passes prompt-only refs
        result = decontaminate_records(
            [
                {
                    "problem_id": "rlvr_1",
                    "family_id": "fam_rlvr",
                    "prompt": prompt,
                    "response": "",
                }
            ],
            reference_texts=[sft_ref_prompt],
            jaccard_reject=1.1,
            jaccard_review=1.0,
            minhash_reject=1.1,
        )[0]
        self.assertEqual(result["status"], "reject")
        self.assertIn("exact_hash_duplicate", result["reasons"])
        # Sanity: the discarded response length is large but irrelevant to identity.
        self.assertGreater(len(long_cot), 500)

    def test_within_batch_exact_prompt_dup_rejected(self) -> None:
        rows = [
            {
                "problem_id": "a",
                "family_id": "f1",
                "prompt": "ما مجموع 2 و 3؟",
                "response": "5",
            },
            {
                "problem_id": "b",
                "family_id": "f2",
                "prompt": "ما مجموع 2 و 3؟",
                "response": "",
            },
        ]
        results = decontaminate_records(
            rows,
            jaccard_reject=1.1,
            jaccard_review=1.0,
            minhash_reject=1.1,
        )
        self.assertEqual(results[0]["status"], "clean")
        self.assertEqual(results[1]["status"], "reject")
        self.assertIn("exact_hash_duplicate", results[1]["reasons"])


if __name__ == "__main__":
    unittest.main()
