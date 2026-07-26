"""Tests for English benchmark ingestion (GSM8K, MATH)."""

from rlvr_pipeline.ingest_english_benchmarks import (
    extract_gsm8k_answer,
    extract_math_answer,
    format_english_benchmark_record,
    convert_gsm8k_dataset,
)


def test_extract_gsm8k_answer():
    text = "Janet had 16 eggs. She used 3 for baking.\n#### 13"
    assert extract_gsm8k_answer(text) == "13"

    with_commas = "The total cost was $1,250 dollars.\n#### 1,250"
    assert extract_gsm8k_answer(with_commas) == "1250"

    no_marker = "The answer is 42"
    assert extract_gsm8k_answer(no_marker) is None


def test_extract_math_answer():
    text = "Simplifying the fraction gives \\boxed{\\frac{3}{4}}."
    assert extract_math_answer(text) == "\\frac{3}{4}"

    simple = "Therefore the result is \\boxed{42}."
    assert extract_math_answer(simple) == "42"


def test_format_english_benchmark_record():
    rec = format_english_benchmark_record(
        question="What is 5 + 7?",
        raw_answer="5 + 7 = 12\n#### 12",
        sample_id="test_1",
        dataset_name="gsm8k",
    )
    assert rec is not None
    assert rec["sample_id"] == "test_1"
    assert rec["domain"] == "gsm8k"
    assert rec["language"] == "en"
    assert rec["ground_truth"] == "12"
    assert rec["answer_spec"]["type"] == "integer"
    assert rec["answer_spec"]["canonical"] == "12"


def test_convert_gsm8k_dataset():
    raw_items = [
        {"question": "What is 10 / 2?", "answer": "10 / 2 = 5\n#### 5"},
        {"question": "What is 3 * 3?", "answer": "3 * 3 = 9\n#### 9"},
        {"question": "Invalid item", "answer": "No answer marker here"},
    ]
    converted = convert_gsm8k_dataset(raw_items)
    assert len(converted) == 2
    assert converted[0]["ground_truth"] == "5"
    assert converted[1]["ground_truth"] == "9"
