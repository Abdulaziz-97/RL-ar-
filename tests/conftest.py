"""Shared test fixtures for the RLVR pipeline."""

import json
from pathlib import Path

import pytest


SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}"
    "{% endfor %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.chat_template = SIMPLE_CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=256,
        vocab_size=tiny_tokenizer.vocab_size,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = tiny_tokenizer.eos_token_id
    return model


def _make_real_rlvr_record(idx, domain, gt, num_steps=3, grade="grade_4"):
    return {
        "id": f"rlvr_{domain}_{idx:04d}",
        "source": "rlvr",
        "domain": domain,
        "prompt": f"What is {idx}+{idx}?",
        "response": "",
        "answer": "",
        "metadata": {
            "id": f"rlvr_{domain}_{idx:04d}",
            "grade_level": grade,
            "num_steps": num_steps,
            "ground_truth_answer": gt,
        },
        "quality": {"score": 1.0, "arabic_purity": 1.0},
    }


@pytest.fixture
def math_jsonl(tmp_path):
    data = [_make_real_rlvr_record(i, "gsm8k", i + i, num_steps=i % 5 + 1) for i in range(1, 9)]
    path = tmp_path / "math.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def logic_jsonl(tmp_path):
    data = [
        _make_real_rlvr_record(i, "logic", f"A={i} B={i+1}", num_steps=3)
        for i in range(1, 5)
    ]
    path = tmp_path / "logic.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def mixed_domain_jsonl(tmp_path):
    data = []
    for i in range(1, 5):
        data.append(_make_real_rlvr_record(i, "gsm8k", i * 2, num_steps=i + 1))
    for i in range(5, 9):
        data.append(_make_real_rlvr_record(i, "math", i * 1.5, num_steps=i - 3))
    for i in range(9, 13):
        data.append(_make_real_rlvr_record(i, "logic", {"X": i}, num_steps=4))
    data.append(_make_real_rlvr_record(13, "mmlu", "", num_steps=0))
    path = tmp_path / "mixed.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def coldstart_jsonl(tmp_path):
    data = [
        {
            "id": f"coldstart_gsm8k_{i:04d}",
            "source": "coldstart",
            "domain": "gsm8k",
            "prompt": f"What is {i}+{i}?",
            "response": f"<think>\nStep 1: {i}+{i}={i+i}\n</think>\n\n#### {i+i}",
            "answer": str(i + i),
            "metadata": {"id": f"coldstart_{i}", "grade_level": "grade_4", "num_steps": 1},
            "quality": {"score": 1.0, "arabic_purity": 0.9},
        }
        for i in range(1, 9)
    ]
    path = tmp_path / "coldstart.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)
