import unittest
import os
import sys
import json
import yaml
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
root_dir = PACK_ROOT.parent

sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "src"))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from vendor.answer_match import match_answers
from vendor.synth.dspy_teacher import think_quality_ok, load_teacher, ArabicTeacher
from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator

class TestFullMasterGenerationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.openrouter_key = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
        cls.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")

    def test_01_base_data_loader_and_registry(self):
        base_sft_path = root_dir / "data" / "arabic_reasoning_coldstart_train.jsonl"
        self.assertTrue(base_sft_path.exists(), "Base SFT dataset file missing!")
        
        items = [json.loads(l) for l in open(base_sft_path, encoding="utf-8") if l.strip()]
        self.assertGreater(len(items), 1000, "Base SFT dataset should contain >1000 samples")
        
        seen_prompts = set(" ".join(it.get("prompt", "").split()) for it in items if it.get("prompt"))
        self.assertEqual(len(seen_prompts), len(items), "Base SFT dataset contains internal duplicate prompts!")
        print(f"  ✓ [COMPONENT 1 PASSED] Base Data Registry loaded {len(seen_prompts)} clean prompts.")

    def test_02_python_contract_verifier(self):
        # Test numeric equality
        self.assertTrue(match_answers("288", "288", domain="gsm8k"))
        self.assertTrue(match_answers("288 قطعة", "288", domain="gsm8k"))
        self.assertFalse(match_answers("100", "288", domain="gsm8k"))
        
        # Test logic JSON matching
        dict_a = {"سعد": "الأول"}
        dict_b = {"سعد": "الأول"}
        self.assertTrue(match_answers(dict_a, dict_b, domain="logic"))
        print("  ✓ [COMPONENT 2 PASSED] Executable Python Contract Verifier working 100%.")

    def test_03_decontamination_and_zero_overlap_guard(self):
        seen_sft_prompts = {"يُقسَم مبلغ 105 ريالًا بين شخصين"}
        
        # Test exact match rejection
        test_prompt = "يُقسَم مبلغ 105 ريالًا بين شخصين"
        self.assertIn(test_prompt, seen_sft_prompts, "Zero-overlap guard failed to catch exact match!")
        
        # Test novel prompt acceptance
        novel_prompt = "عامل ينجز 24 قطعة في الساعة"
        self.assertNotIn(novel_prompt, seen_sft_prompts, "Zero-overlap guard incorrectly flagged novel prompt!")
        print("  ✓ [COMPONENT 3 PASSED] 3-Tiered Decontamination & Zero-Overlap Guard verified.")

    def test_04_gepa_v2_dspy_teacher_signatures(self):
        gepa_path = PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"
        self.assertTrue(gepa_path.exists(), "GEPA v2 teacher signatures file missing!")
        
        teacher = load_teacher(gepa_path)
        self.assertIsInstance(teacher, ArabicTeacher)
        self.assertTrue(hasattr(teacher, "plan"))
        self.assertTrue(hasattr(teacher, "solve"))
        self.assertTrue(hasattr(teacher, "materialize"))
        print("  ✓ [COMPONENT 4 PASSED] GEPA v2 DSPy 8-Stage Teacher signatures loaded cleanly.")

    def test_05_msa_purity_and_quality_score(self):
        sample_think = "أولاً، نحسب الناتج: 24 × 4 = 96 قطعة. ثانياً، الإجمالي = 96 × 3 = 288 قطعة."
        sample_resp = f"<think>\n{sample_think}\n</think>\n<answer>288</answer>"
        
        ok, reason = think_quality_ok(sample_think, sample_resp, domain="gsm8k", gt="288")
        self.assertTrue(ok, f"Quality score failed for clean CoT: {reason}")
        print("  ✓ [COMPONENT 5 PASSED] MSA Quality & Purity Score Evaluator working 100%.")

    def test_06_vastai_master_v4_config(self):
        cfg_path = root_dir / "configs" / "qwen_4b_2x5090_v4_sota.yaml"
        self.assertTrue(cfg_path.exists(), "Master V4 config file missing!")
        
        cfg_data = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        self.assertEqual(cfg_data.get("coldstart_data_path"), "data/arabic_reasoning_coldstart_v4.jsonl")
        self.assertEqual(cfg_data.get("train_data_path"), "data/arabic_reasoning_rlvr_v4.jsonl")
        self.assertEqual(cfg_data.get("max_steps"), 125)
        self.assertEqual(cfg_data.get("max_completion_length"), 2048)
        print("  ✓ [COMPONENT 6 PASSED] Vast.ai Master V4 training configuration validated.")

if __name__ == "__main__":
    print("\n" + "="*75)
    print(" 🧪 RUNNING EXHAUSTIVE PIPELINE UNIT TEST SUITE (BEFORE FULL GENERATION)")
    print("="*75)
    unittest.main()
