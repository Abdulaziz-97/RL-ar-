import unittest
import sys
from pathlib import Path

# Add official_eval to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "official_eval"))

from tasks.araeval.generative_utils import extract_answer, grade_answer, strip_thinking_tags
from tasks.araeval.utils import normalize_aramath, normalize_arapro, normalize_araifeval, process_ifeval_results

class TestGenerativeExtraction(unittest.TestCase):

    def test_strip_thinking_tags_closed(self):
        text = "<think>\nThis is a long calculation: 15 * 4 = 60\n</think>\nالإجابة: A"
        self.assertEqual(strip_thinking_tags(text).strip(), "الإجابة: A")

    def test_strip_thinking_tags_unclosed(self):
        text = "<think>\nThis is a long calculation without closing tag..."
        self.assertEqual(strip_thinking_tags(text).strip(), "")

    def test_strip_thinking_tags_prepended(self):
        text = "Calculations inside think tag...</think>\nالإجابة هي (ب)"
        self.assertEqual(strip_thinking_tags(text).strip(), "الإجابة هي (ب)")

    def test_extract_answer_closed_think_latin(self):
        text = "<think>15 + 20 = 35</think>\nAnswer: B"
        self.assertEqual(extract_answer(text), "B")

    def test_extract_answer_closed_think_arabic(self):
        text = "<think>نقوم بحساب المعادلة...</think>\nالإجابة: ج"
        self.assertEqual(extract_answer(text), "C")

    def test_extract_answer_arabic_option_statement(self):
        text = "<think>المرتبة الأولى هي الصحيحة</think>\nالخيار الصحيح هو (أ)"
        self.assertEqual(extract_answer(text), "A")

    def test_extract_answer_numeric_digit(self):
        text = "<think>الحل النهائي يساوي 2</think>\nالإجابة: 2"
        self.assertEqual(extract_answer(text), "B")

    def test_extract_answer_eastern_arabic_digit(self):
        text = "<think>الحل النهائي يساوي ٣</think>\nالخيار هو (٣)"
        self.assertEqual(extract_answer(text), "C")

    def test_extract_answer_inside_unclosed_think(self):
        text = "<think>أولاً نحسب الناتج 50 * 2 = 100. إذن الخيار الصحيح هو B"
        self.assertEqual(extract_answer(text), "B")

    def test_extract_answer_markdown_brackets(self):
        text = "وبالتالي نحصل على **D** كخيار مناسب"
        self.assertEqual(extract_answer(text), "D")

    def test_grade_answer(self):
        self.assertTrue(grade_answer("A", 0))
        self.assertTrue(grade_answer("B", 1))
        self.assertTrue(grade_answer("C", 2))
        self.assertTrue(grade_answer("D", 3))
        self.assertFalse(grade_answer("A", 1))
        self.assertFalse(grade_answer(None, 0))

    def test_ifeval_grading(self):
        row = {
            "key": 1,
            "prompt": "اكتب مقالاً وأضف عنواناً بين <<علامات تنصيص>>.",
            "categories": ["title"],
        }
        norm = normalize_araifeval(row)
        self.assertIn("prompt", norm)
        
        # Test valid response containing <<عنوان>>
        response = "<<مستقبل الذكاء الاصطناعي في المملكة>>\nهذا هو نص المقال."
        res = process_ifeval_results(norm, [response])
        self.assertTrue(res.get("prompt_level_strict_acc"))

if __name__ == "__main__":
    unittest.main()
