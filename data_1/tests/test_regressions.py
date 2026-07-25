from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor")]

from rlvr_contracts.response import parse_response
from rlvr_contracts.answer_spec import (
    AnswerSpecError,
    canonicalize_decimal_exact,
    parse_answer_spec,
)
from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator, _read_jsonl, _write_jsonl
from rlvr_synth.calibration.pass_at_n import assign_band, calibrate_pass_at_n
from rlvr_synth.calibration.tranches import merge_tranches
from rlvr_synth.decontam.pipeline import decontaminate_records
from rlvr_synth.release.manifest import ReleaseManifest
from rlvr_synth.release.ship_gate import audit_family_splits, run_ship_gate
from rlvr_synth.roles.protocols import RenderedProblem
import zlib

from backends.dspy_backend import (
    ContractVerifier,
    LiveTraceTeacher,
)

def _stable_int(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))
from synth.dspy_teacher import match_gt, think_has_bare_ops
from synth.programmatic import (
    _hard_bus_trip_budget,
    _hard_factory_shipment,
    _hard_farm_harvest,
    _hard_store_restock,
    think_leaks_final_gt,
)
from vendor.answer_match import answers_match_numeric
from synth.dspy_teacher import BudgetCallback, BudgetState, make_lm


class ResponseGrammarTests(unittest.TestCase):
    def test_requires_think_before_answer_and_no_extra_text(self) -> None:
        valid = "<think>\nخطوة أولى ثم خطوة ثانية واضحة\n</think>\n<answer>1</answer>"
        reversed_blocks = "<answer>1</answer><think>خطوة أولى ثم خطوة ثانية واضحة</think>"
        prefixed = "noise " + valid
        self.assertTrue(parse_response(valid).format_ok)
        self.assertFalse(parse_response(reversed_blocks).format_ok)
        self.assertFalse(parse_response(prefixed).format_ok)


class AnswerSpecRegressionTests(unittest.TestCase):
    def test_decimal_exact_does_not_round_small_float_to_zero(self) -> None:
        self.assertEqual(canonicalize_decimal_exact(1e-7), "0.0000001")

    def test_tolerance_must_be_non_negative_and_approx_only(self) -> None:
        with self.assertRaises(AnswerSpecError):
            parse_answer_spec(
                {"type": "decimal_approx", "canonical": "1.0", "tolerance": -1}
            )
        with self.assertRaises(AnswerSpecError):
            parse_answer_spec(
                {"type": "integer", "canonical": "1", "tolerance": 0.1}
            )


class OrchestratorRegressionTests(unittest.TestCase):
    def test_invalid_generation_config_fails_early(self) -> None:
        with self.assertRaises(ValueError):
            SynthOrchestrator(SynthConfig(n_families=1, domains=[]))
        with self.assertRaises(ValueError):
            SynthOrchestrator(SynthConfig(partitions={}))
        with self.assertRaises(ValueError):
            SynthOrchestrator(SynthConfig(partitions={"sft_train": -1.0}))

    def test_same_answer_different_prompts_are_not_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orch = SynthOrchestrator(SynthConfig(n_families=0, work_dir=tmp))
            rows = [
                {
                    "family_id": "a",
                    "domain": "math",
                    "partition": "sft_train",
                    "answer_spec": {"type": "integer", "canonical": "0"},
                    "latent": {"prompt": "احسب 1 - 1"},
                    "seed": 1,
                },
                {
                    "family_id": "b",
                    "domain": "math",
                    "partition": "sft_train",
                    "answer_spec": {"type": "integer", "canonical": "0"},
                    "latent": {"prompt": "احسب 2 - 2"},
                    "seed": 2,
                },
            ]
            _write_jsonl(orch.stages_dir / "latent_spec.jsonl", rows)
            orch.stage_deterministic_solve()
            got = _read_jsonl(orch.stages_dir / "deterministic_solve.jsonl")
            self.assertEqual([row["unique"] for row in got], [True, True])

    def test_rlvr_partition_skips_teacher_and_releases_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SynthConfig(
                n_families=4,
                traces_per_problem=2,
                work_dir=tmp,
                backend="stub",
                partitions={"rlvr_train": 1.0},
            )
            orch = SynthOrchestrator(cfg)
            orch.backend.trace_teacher.sample_traces = lambda *args, **kwargs: self.fail(
                "RLVR must not call the trace teacher"
            )
            orch.run(resume=False)
            rows = _read_jsonl(Path(tmp) / "release_corpora" / "rlvr_train.jsonl")
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["response"] == "" for row in rows))

    def test_fresh_run_clears_managed_outputs_and_resume_checks_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SynthConfig(n_families=4, work_dir=tmp, backend="stub")
            orch = SynthOrchestrator(cfg)
            orch.run(resume=False)
            first_quarantine = _read_jsonl(orch.quarantine_path)
            orch.run(resume=False)
            self.assertEqual(_read_jsonl(orch.quarantine_path), first_quarantine)

            changed = SynthConfig(n_families=5, work_dir=tmp, backend="stub")
            with self.assertRaises(RuntimeError):
                SynthOrchestrator(changed).run(resume=True)

    def test_unselected_valid_alternate_is_not_silently_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orch = SynthOrchestrator(
                SynthConfig(n_families=0, work_dir=tmp, max_alternate_methods=0)
            )
            rows = [
                {
                    "problem_id": "p",
                    "family_id": "f",
                    "partition": "sft_train",
                    "domain": "math",
                    "prompt": "سؤال",
                    "answer_spec": {"type": "integer", "canonical": "1"},
                    "response": "<think>حل طويل بما يكفي للقبول هنا</think><answer>1</answer>",
                    "method_id": method,
                    "teacher": "stub",
                    "concision_tokens": tokens,
                    "gate_pass": True,
                }
                for method, tokens in (("primary", 10), ("alternate", 20))
            ]
            _write_jsonl(orch.stages_dir / "gates.jsonl", rows)
            orch.stage_select_quarantine()
            selected = _read_jsonl(orch.stages_dir / "selected.jsonl")
            self.assertEqual([row["method_id"] for row in selected], ["primary"])


class ReleaseAndCalibrationTests(unittest.TestCase):
    def test_ship_split_audit_includes_within_track_train_eval(self) -> None:
        gates = audit_family_splits(
            {
                "sft_train": [{"family_id": "same", "prompt": "أ"}],
                "sft_eval": [{"family_id": "same", "prompt": "ب"}],
            }
        )
        self.assertTrue(
            any(
                gate.name == "family_overlap:sft_eval|sft_train"
                and not gate.passed
                for gate in gates
            )
        )

    def test_decontamination_scans_response_as_well_as_prompt(self) -> None:
        result = decontaminate_records(
            [
                {
                    "problem_id": "p",
                    "family_id": "f",
                    "prompt": "سؤال نظيف",
                    "response": "This came from gsm8k",
                }
            ]
        )[0]
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("benchmark_registry", result["reasons"])

    def test_reference_minhash_hit_is_rejected(self) -> None:
        with patch(
            "rlvr_synth.decontam.pipeline.LSHIndex.query",
            return_value={"ref:0"},
        ), patch(
            "rlvr_synth.decontam.pipeline.MinHashSketch.estimate_jaccard",
            return_value=0.95,
        ):
            result = decontaminate_records(
                [{"problem_id": "p", "family_id": "f", "prompt": "different"}],
                reference_texts=["reference"],
                jaccard_reject=1.1,
                jaccard_review=1.0,
                minhash_reject=0.9,
            )[0]
        self.assertEqual(result["status"], "reject")
        self.assertIn("minhash_reference", result["reasons"])

    def test_manifest_verifier_version_mismatch_fails_ship_gate(self) -> None:
        manifest = ReleaseManifest(
            release_id="r",
            created_at="now",
            corpus_version="v",
            verifier_registry_version="wrong-version",
        )
        report = run_ship_gate(corpora={}, manifest=manifest)
        self.assertFalse(report.passed)
        self.assertTrue(
            any(g.name == "verifier_registry_match" and not g.passed for g in report.gates)
        )

    def test_tranche_cache_includes_source_order_and_dedupe_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            _write_jsonl(first, [{"problem_id": "same", "value": 1}])
            _write_jsonl(second, [{"problem_id": "same", "value": 2}])
            out = root / "out.jsonl"
            merge_tranches([first, second], out, dedupe_on="problem_id")
            self.assertEqual(_read_jsonl(out)[0]["value"], 1)
            result = merge_tranches([second, first], out, dedupe_on="problem_id")
            self.assertFalse(result.cache_hit)
            self.assertEqual(_read_jsonl(out)[0]["value"], 2)

    def test_pass_at_n_rejects_invalid_n_and_fraction(self) -> None:
        with self.assertRaises(ValueError):
            assign_band(float("nan"))
        with self.assertRaises(ValueError):
            calibrate_pass_at_n([], generate_fn=lambda p, s: "", verify_fn=lambda p, c: False, n=0)


class ProviderTests(unittest.TestCase):
    def test_registered_provider_ignores_stale_generic_credentials(self) -> None:
        spec = get_provider("openai")
        with patch.dict(
            os.environ,
            {
                "RLVR_API_KEY": "stale-generic",
                "RLVR_API_BASE": "https://wrong.invalid",
                "OPENAI_API_KEY": "provider-key",
            },
            clear=True,
        ):
            self.assertEqual(resolve_api_key(spec), "provider-key")
            self.assertEqual(resolve_api_base(spec), spec.api_base)

    def test_live_ids_use_stable_hashing(self) -> None:
        self.assertEqual(_stable_int("math"), _stable_int("math"))
        self.assertEqual(_stable_int("math"), 1487328896)

    def test_zero_requested_live_traces_makes_no_teacher_call(self) -> None:
        teacher = LiveTraceTeacher.__new__(LiveTraceTeacher)
        teacher.teacher = lambda **kwargs: self.fail("teacher must not be called")
        problem = RenderedProblem(
            problem_id="p",
            family_id="f",
            domain="math",
            partition="sft_train",
            prompt="سؤال",
            answer_spec={"type": "integer", "canonical": "1"},
        )
        self.assertEqual(teacher.sample_traces(problem, 0, 0), [])

    def test_positive_live_trace_uses_dspy_context(self) -> None:
        teacher = LiveTraceTeacher.__new__(LiveTraceTeacher)
        teacher.provider = "test"
        teacher.model = "test-model"
        teacher.lm = dspy.LM(
            model="openai/test",
            api_key="test",
            api_base="https://example.invalid/v1",
        )
        teacher.budget = BudgetState(Path(tempfile.gettempdir()) / "unused.json", 1.0)
        teacher.teacher = lambda **kwargs: dspy.Prediction(
            response=(
                "<think>\nنقرأ المسألة بعناية.\nنحسب القيمة المطلوبة بوضوح.\n"
                "</think>\n<answer>1</answer>"
            )
        )
        problem = RenderedProblem(
            problem_id="p",
            family_id="f",
            domain="math",
            partition="sft_train",
            prompt="سؤال",
            answer_spec={
                "type": "integer",
                "canonical": "1",
                "ground_truth_structured": 1,
            },
        )
        traces = teacher.sample_traces(problem, 1, 10)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].metadata["seed"], 10)

    def test_budgeted_lm_rollout_clone_does_not_copy_thread_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetState(Path(tmp) / "budget.json", 1.0)
            lm = dspy.LM(
                model="openai/test",
                api_key="test",
                callbacks=[BudgetCallback(budget)],
            )
            clone = _lm_for_rollout(lm, 7)
            self.assertEqual(clone.kwargs["rollout_id"], 7)
            self.assertIs(clone.callbacks[0].budget, budget)

    def test_teacher_matchers_fail_closed(self) -> None:
        self.assertFalse(answers_match_numeric("42 is wrong", 42))
        self.assertFalse(match_gt("logic", "color red size 2", {"color": "red", "size": 2}))
        self.assertTrue(think_has_bare_ops("نحسب 6 مضروبا في 7."))

    def test_common_answer_assertions_are_detected_as_leaks(self) -> None:
        self.assertTrue(think_leaks_final_gt("الجواب هو 152", 152, "math"))
        self.assertTrue(think_leaks_final_gt("اذن هو 42", 42, "math"))

    def test_equal_distribution_generators_emit_valid_equations(self) -> None:
        generators = (
            _hard_factory_shipment,
            _hard_farm_harvest,
            _hard_store_restock,
            _hard_bus_trip_budget,
        )
        for generator in generators:
            for seed in range(50):
                sample = generator(random.Random(seed))
                self.assertEqual(
                    _invalid_arithmetic_equations("\n".join(sample.solution_steps)),
                    [],
                    (generator.__name__, seed),
                )

    def test_explicit_false_arithmetic_is_rejected(self) -> None:
        self.assertEqual(_invalid_arithmetic_equations("نحسب ٢ + ٣ = ٦"), ["2 + 3 = 6"])
        self.assertEqual(_invalid_arithmetic_equations("13 + 13 + 13 = 39"), [])
        verifier = ContractVerifier()
        problem = RenderedProblem(
            problem_id="p",
            family_id="f",
            domain="math",
            partition="sft_train",
            prompt="سؤال",
            answer_spec={"type": "integer", "canonical": "5"},
        )
        from rlvr_synth.roles.protocols import TraceCandidate

        trace = TraceCandidate(
            problem_id="p",
            response=(
                "<think>\nنبدأ بالحساب المطلوب.\nنحسب ٢ + ٣ = ٦ ثم نراجع.\n"
                "</think>\n<answer>5</answer>"
            ),
            method_id="m",
            teacher="t",
        )
        ok, reasons = verifier.verify_steps(problem, trace)
        self.assertFalse(ok)
        self.assertTrue(any(reason.startswith("invalid_arithmetic") for reason in reasons))

    def test_small_budget_is_not_disabled_by_fixed_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetState(Path(tmp) / "budget.json", 0.10)
            self.assertTrue(budget.ok())

    def test_budget_records_final_paid_call_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetState(Path(tmp) / "budget.json", 0.10)
            callback = BudgetCallback(
                budget, price_in=0.001, price_out=0.001
            )
            fake_lm = SimpleNamespace(
                history=[
                    {
                        "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                        "model": "test",
                    }
                ],
                model="test",
            )
            with dspy.context(lm=fake_lm):
                with self.assertRaisesRegex(RuntimeError, "BUDGET_CAP_HIT"):
                    callback.on_lm_end("call-1", object())
            self.assertEqual(budget.data["calls"], 1)
            self.assertTrue(budget.data["cap_hit"])

    def test_missing_usage_marks_budget_accounting_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetState(Path(tmp) / "budget.json", 1.0)
            callback = BudgetCallback(budget)
            fake_lm = SimpleNamespace(
                history=[{"usage": {}, "model": "test"}],
                model="test",
            )
            with dspy.context(lm=fake_lm):
                callback.on_lm_end("call-no-usage", object())
            self.assertTrue(budget.data["usage_missing"])

    def test_unknown_pricing_fails_closed_when_budgeted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetState(Path(tmp) / "budget.json", 1.0)
            with patch.dict(os.environ, {"RLVR_API_KEY": "test"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "pricing is unknown"):
                    make_lm(
                        provider="openrouter",
                        model="vendor/model",
                        budget=budget,
                    )


if __name__ == "__main__":
    unittest.main()
