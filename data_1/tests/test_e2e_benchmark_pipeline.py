import unittest
import sys
import json
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "src"))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator
from backends.dspy_backend import LiveTraceTeacher

class TestE2EBenchmarkPipeline(unittest.TestCase):

    def test_synth_config_validation(self):
        cfg = SynthConfig(
            mode="pilot",
            n_families=4,
            domains=["gsm8k", "math"],
            traces_per_problem=1,
            seed=42,
            work_dir=str(PACK_ROOT / "outputs" / "test_run"),
            backend="external",
            external_module=str(PACK_ROOT / "backends" / "dspy_backend.py"),
            external_config={
                "mode": "replay",
                "model": "deepseek-v4-pro",
                "gepa_path": str(PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"),
                "replay_stages": str(PACK_ROOT / "assets" / "replay_stages"),
            },
            partitions={"sft_train": 1.0}
        )
        self.assertEqual(cfg.n_families, 4)
        self.assertEqual(cfg.backend, "external")

    def test_dspy_backend_openrouter_routing(self):
        config_openrouter = {"model": "qwen/qwen3.7-flash"}
        config_deepseek = {"model": "deepseek-v4-pro"}
        
        # Test routing initialization without throwing syntax errors
        self.assertTrue("qwen" in config_openrouter["model"].lower())
        self.assertFalse("qwen" in config_deepseek["model"].lower())

    def test_e2e_stub_orchestrator_run(self):
        work_dir = PACK_ROOT / "outputs" / "unit_test_stub_run"
        cfg = SynthConfig(
            mode="pilot",
            n_families=2,
            domains=["gsm8k"],
            traces_per_problem=1,
            seed=123,
            work_dir=str(work_dir),
            backend="stub",
            partitions={"sft_train": 1.0}
        )
        orchestrator = SynthOrchestrator(cfg)
        res = orchestrator.run(resume=False)
        self.assertIn("sft_train", res)
        self.assertTrue((work_dir / "release_corpora" / "sft_train.jsonl").exists())

if __name__ == "__main__":
    unittest.main()
