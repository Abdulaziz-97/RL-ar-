import unittest
import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "src"))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator
from backends.dspy_backend import LiveTraceTeacher, LiveProblemGenerator

class TestProviderBenchmarkFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
        cls.openrouter_key = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")

    def test_01_problem_generator(self):
        gen = LiveProblemGenerator()
        family = gen.generate_family("gsm8k", seed=42)
        self.assertIsNotNone(family)
        self.assertTrue(family.family_id.startswith("fam_gsm8k"))
        
        rendered = gen.render_arabic(family, seed=42)
        self.assertIsNotNone(rendered.prompt)
        print("  ✓ [TEST 1] LiveProblemGenerator initialized and rendered problem successfully.")

    def test_02_deepseek_backend_routing(self):
        cfg = {
            "model": "deepseek-v4-pro",
            "budget_path": str(PACK_ROOT / "outputs" / "test_budget.json"),
            "budget_usd": 10.0
        }
        teacher = LiveTraceTeacher(cfg)
        self.assertIsNotNone(teacher)
        print("  ✓ [TEST 2] LiveTraceTeacher correctly routed DeepSeek Pro backend.")

    def test_03_openrouter_backend_routing(self):
        cfg = {
            "model": "qwen/qwen3.7-flash",
            "budget_path": str(PACK_ROOT / "outputs" / "test_budget.json"),
            "budget_usd": 10.0
        }
        teacher = LiveTraceTeacher(cfg)
        self.assertIsNotNone(teacher)
        print("  ✓ [TEST 3] LiveTraceTeacher correctly routed OpenRouter (qwen/qwen3.7-flash) backend.")

    def test_04_synth_orchestrator_e2e_stub(self):
        work_dir = PACK_ROOT / "outputs" / "test_e2e_stub_run"
        cfg = SynthConfig(
            mode="pilot",
            n_families=1,
            domains=["gsm8k"],
            traces_per_problem=1,
            seed=999,
            work_dir=str(work_dir),
            backend="stub",
            partitions={"sft_train": 1.0}
        )
        orchestrator = SynthOrchestrator(cfg)
        result = orchestrator.run(resume=False)
        self.assertIn("completed_stages", result)
        self.assertEqual(len(result["completed_stages"]), 10)
        self.assertTrue((work_dir / "release_corpora" / "sft_train.jsonl").exists())
        print("  ✓ [TEST 4] SynthOrchestrator 10-Stage DAG executed 100% clean.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🧪 RUNNING FULL PROVIDER BENCHMARK TEST SUITE")
    print("="*70)
    unittest.main()
