"""Unit tests for newly added codebase scripts and utilities.

Tests:
  - scripts/filter_hard_dataset.py (derive_difficulty heuristic & JSONL filtering)
  - scripts/run_benchmark_probe.py (load_probe_dataset fixed sampling & config validation)
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.filter_hard_dataset import derive_difficulty
from scripts.run_benchmark_probe import PROBE_CONFIGS, load_probe_dataset


class TestFilterHardDatasetScript(unittest.TestCase):
    """Test data curation and filtering rules in scripts/filter_hard_dataset.py."""

    def test_derive_difficulty_trivial_pass8(self):
        rec = {"metadata": {"base_pass_8": 0.875}}
        tag, keep = derive_difficulty(rec)
        self.assertEqual(tag, "trivial")
        self.assertFalse(keep)

    def test_derive_difficulty_num_steps_trivial(self):
        rec = {"metadata": {"num_steps": 2, "grade_level": "grade_3"}}
        tag, keep = derive_difficulty(rec)
        self.assertEqual(tag, "trivial")
        self.assertFalse(keep)

    def test_derive_difficulty_medium(self):
        rec = {"metadata": {"num_steps": 4, "grade_level": "grade_5"}}
        tag, keep = derive_difficulty(rec)
        self.assertEqual(tag, "medium")
        self.assertTrue(keep)

    def test_derive_difficulty_hard(self):
        rec = {"metadata": {"num_steps": 7, "grade_level": "grade_8"}}
        tag, keep = derive_difficulty(rec)
        self.assertEqual(tag, "hard")
        self.assertTrue(keep)

    def test_full_file_filtering(self):
        sample_data = [
            {"id": 1, "metadata": {"num_steps": 2, "grade_level": "grade_1"}}, # Trivial -> Drop
            {"id": 2, "metadata": {"num_steps": 3, "grade_level": "grade_5"}}, # Medium -> Keep
            {"id": 3, "metadata": {"num_steps": 6, "grade_level": "grade_8"}}, # Hard -> Keep
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp_in = f.name

        tmp_out = tmp_in + "_out.jsonl"

        try:
            records = []
            with open(tmp_in, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line.strip()))

            kept = []
            for rec in records:
                tag, keep = derive_difficulty(rec)
                if keep:
                    rec["difficulty_tag"] = tag
                    kept.append(rec)

            self.assertEqual(len(kept), 2)
            self.assertEqual(kept[0]["id"], 2)
            self.assertEqual(kept[0]["difficulty_tag"], "medium")
            self.assertEqual(kept[1]["id"], 3)
            self.assertEqual(kept[1]["difficulty_tag"], "hard")
        finally:
            Path(tmp_in).unlink(missing_ok=True)
            Path(tmp_out).unlink(missing_ok=True)


class TestRunBenchmarkProbeScript(unittest.TestCase):
    """Test benchmark probe telemetry script features in scripts/run_benchmark_probe.py."""

    def test_probe_configs_structure(self):
        self.assertIn("araeval_aramath", PROBE_CONFIGS)
        self.assertIn("araeval_arapro", PROBE_CONFIGS)

        aramath_cfg = PROBE_CONFIGS["araeval_aramath"]
        self.assertIsNone(aramath_cfg["sample_size"])  # Full 605

        arapro_cfg = PROBE_CONFIGS["araeval_arapro"]
        self.assertEqual(arapro_cfg["sample_size"], 500)  # Fixed 500 sample

    @patch("datasets.load_dataset")
    def test_load_probe_dataset_sampling(self, mock_load_dataset):
        # Create 1000 dummy documents
        dummy_ds = [{"question": f"q_{i}", "answer": f"a_{i}"} for i in range(1000)]
        mock_load_dataset.return_value = dummy_ds

        dummy_normalizer = lambda doc: {"target": doc["answer"], "question": doc["question"]}

        config_sampled = {
            "path": "dummy/path",
            "revision": "main",
            "split": "test",
            "normalizer": dummy_normalizer,
            "sample_size": 100,
        }

        docs = load_probe_dataset("dummy_task", config_sampled)
        self.assertEqual(len(docs), 100)

        # Deterministic sampling check (seed=42)
        docs_again = load_probe_dataset("dummy_task", config_sampled)
        self.assertEqual(docs, docs_again)


class TestBenchmarkProbeCallback(unittest.TestCase):
    """Test automatic benchmark probe callback in isolated subprocess mode."""

    def test_hard_math_augmentation(self):
        from scripts.mine_and_augment_hard_math import augment_questions
        sample_questions = [
            {"question": "مستطيل طوله 38 مترًا وعرضه 26 مترًا. كم مساحته؟", "gold": "988"},
            {"question": "اشترى تاجر 15 قطعة بمبلغ 525 ريالًا. ما سعر القطعة؟", "gold": "35"}
        ]
        augmented = augment_questions(sample_questions, target_count=10)
        assert len(augmented) == 10
        assert "<think>" in augmented[0]["solution"]
        assert "</answer>" in augmented[0]["solution"]
        assert "988" in augmented[0]["solution"]

    @patch("subprocess.run")
    def test_on_save_triggers_probe_subprocess(self, mock_subprocess_run):
        from rlvr_pipeline.benchmark_probe_callback import BenchmarkProbeCallback
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="Probe Success")

        cb = BenchmarkProbeCallback(enable_probe=True)

        mock_args = MagicMock()
        mock_args.process_index = 0
        mock_args.output_dir = "./outputs"
        mock_args.report_to = "none"

        mock_state = MagicMock()
        mock_state.global_step = 50

        cb.on_save(mock_args, mock_state, MagicMock())
        mock_subprocess_run.assert_called_once()
        
        # Verify DDP env vars were stripped from subprocess env
        _, kwargs = mock_subprocess_run.call_args
        env_used = kwargs.get("env", {})
        self.assertNotIn("MASTER_PORT", env_used)
        self.assertNotIn("MASTER_ADDR", env_used)


class TestArabicTerminalRenderer(unittest.TestCase):
    """Test Arabic text formatting and BiDi reshaping for terminal display."""

    def test_format_arabic_terminal_text_basic(self):
        from rlvr_pipeline.utils import format_arabic_terminal_text
        arabic_sample = "الالتزام بها بالضبط: 1. ابدأ بـ: <think>"
        result = format_arabic_terminal_text(arabic_sample)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_format_arabic_terminal_text_strips_box_borders(self):
        from rlvr_pipeline.utils import format_arabic_terminal_text
        corrupted_box = "│ في الحساب الضروي هو │ 610 ريالاً │"
        cleaned = format_arabic_terminal_text(corrupted_box)
        self.assertNotIn("│", cleaned)

    def test_format_arabic_terminal_text_empty(self):
        from rlvr_pipeline.utils import format_arabic_terminal_text
        self.assertEqual(format_arabic_terminal_text(""), "")


if __name__ == "__main__":
    unittest.main()
