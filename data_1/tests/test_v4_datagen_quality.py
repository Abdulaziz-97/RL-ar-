"""Unit/integration tests for V4 dual-corpus datagen quality bar."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor")]

from rlvr_synth.quality.ids import content_family_id, content_problem_id
from rlvr_synth.quality.quotas import (
    DOMAIN_QUOTAS,
    RLVR_BAND_MATRIX,
    select_domain_quota,
    select_rlvr_band_matrix,
)
from rlvr_synth.calibration.pass_at_n import assign_band, calibrate_pass_at_n, stub_generate_fn_factory
from rlvr_synth.release.ship_gate import run_ship_gate
from synth.programmatic import gen_gsm8k, gen_logic, gen_math, gen_math_comp
import random


class StableIdTests(unittest.TestCase):
    def test_content_ids_are_stable(self) -> None:
        a = content_problem_id(domain="gsm8k", prompt="س 1", answer_canonical="3", seed=1)
        b = content_problem_id(domain="gsm8k", prompt="س 1", answer_canonical="3", seed=1)
        c = content_problem_id(domain="gsm8k", prompt="س 2", answer_canonical="3", seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        f1 = content_family_id(domain="math", prompt="س", template_family="math_percent", seed=2)
        f2 = content_family_id(domain="math", prompt="س", template_family="math_percent", seed=2)
        self.assertEqual(f1, f2)


class TemplateFamilyTests(unittest.TestCase):
    def test_each_core_domain_exposes_structural_template_family(self) -> None:
        rng = random.Random(0)
        samples = [gen_gsm8k(rng), gen_math(rng), gen_math_comp(rng), gen_logic(rng)]
        for s in samples:
            fam = (s.meta_extra or {}).get("template_family")
            self.assertTrue(fam)
            self.assertNotEqual(fam, s.domain)

    def test_at_least_eight_families_per_domain_reachable(self) -> None:
        for domain, gen in {
            "gsm8k": gen_gsm8k,
            "math": gen_math,
            "math_comp": gen_math_comp,
            "logic": gen_logic,
        }.items():
            seen = set()
            rng = random.Random(123)
            for _ in range(400):
                seen.add((gen(rng).meta_extra or {}).get("template_family"))
            self.assertGreaterEqual(len(seen), 8, msg=f"{domain} families={seen}")


class QuotaSelectionTests(unittest.TestCase):
    def test_domain_quota_reports_shortage(self) -> None:
        rows = [{"domain": "gsm8k", "problem_id": f"p{i}", "family_id": f"f{i}"} for i in range(10)]
        selected, shortages = select_domain_quota(rows, quotas={"gsm8k": 12, "math": 1})
        self.assertEqual(len(selected), 10)
        self.assertEqual(shortages["gsm8k"], 2)
        self.assertEqual(shortages["math"], 1)

    def test_rlvr_matrix_fills_exact_counts(self) -> None:
        rows = []
        n = 0
        for domain, bands in RLVR_BAND_MATRIX.items():
            for band, need in bands.items():
                for i in range(need + 2):  # surplus
                    n += 1
                    emp_band = "hard" if band == "hard_diagnostic" else band
                    passes = 1 if band == "hard_diagnostic" else {"easy": 6, "medium": 4, "hard": 2}[emp_band]
                    rows.append(
                        {
                            "domain": domain,
                            "problem_id": f"{domain}_{band}_{i}",
                            "family_id": f"fam_{domain}_{band}_{i}",
                            "empirical_difficulty": {
                                "band": emp_band,
                                "passes": passes,
                                "n": 8,
                                "pass_fraction": passes / 8,
                            },
                        }
                    )
        selected, mastered, deferred, shortages = select_rlvr_band_matrix(rows)
        self.assertEqual(shortages, {})
        self.assertEqual(len(selected), 4000)
        self.assertEqual(len(mastered), 0)
        self.assertEqual(sum(DOMAIN_QUOTAS.values()), 4000)


class PassAtNWiringTests(unittest.TestCase):
    def test_stub_calibration_writes_bands(self) -> None:
        problems = [
            {
                "problem_id": "p1",
                "prompt": "ما مجموع 2 و 3؟",
                "answer_spec": {"type": "integer", "canonical": "5"},
            }
        ]

        def verify_fn(prob, completion: str) -> bool:
            return "<answer>5</answer>" in completion

        results = calibrate_pass_at_n(
            problems,
            generate_fn=stub_generate_fn_factory({"ما مجموع 2 و 3؟": "5"}),
            verify_fn=verify_fn,
            n=8,
            base_seed=0,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].n, 8)
        self.assertEqual(len(set(results[0].seeds)), 8)
        self.assertEqual(results[0].band, assign_band(results[0].pass_fraction))


class ShipGateStrictOptInTests(unittest.TestCase):
    def test_exact_count_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sft.jsonl"
            rows = []
            for i in range(2):
                rows.append(
                    {
                        "problem_id": f"p{i}",
                        "family_id": f"f{i}",
                        "partition": "sft_train",
                        "domain": "gsm8k",
                        "prompt": f"سؤال {i} مع تفاصيل كافية للتحقق",
                        "response": "<think>\nخطوة أولى واضحة ثم خطوة ثانية واضحة ثم ناتج\n</think>\n<answer>1</answer>",
                        "answer_spec": {"type": "integer", "canonical": "1"},
                        "verifier_result": True,
                        "verifier_type": "python",
                        "verifier_version": "rlvr-contracts-verifiers-v1",
                        "trace_audit": {
                            "format_ok": True,
                            "step_verified": True,
                            "concision_tokens": 10,
                        },
                        "metadata": {"decontam": {"status": "clean"}},
                    }
                )
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            report = run_ship_gate(
                corpora={"sft_train": path},
                schema_kind_by_corpus={"sft_train": "sft_trace"},
                require_exact_counts={"sft_train": 2},
                require_decontam_clean=True,
            )
            self.assertTrue(
                any(g.name == "exact_count:sft_train" and g.passed for g in report.gates)
            )


class QwenDemotionTests(unittest.TestCase):
    def test_live_generator_refuses_without_experimental_flag(self) -> None:
        import os
        import subprocess

        env = os.environ.copy()
        env.pop("EXPERIMENTAL_QWEN_DATAGEN", None)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_qwen37_rlvr_live.py")],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("demoted", (proc.stderr + proc.stdout).lower())


if __name__ == "__main__":
    unittest.main()
