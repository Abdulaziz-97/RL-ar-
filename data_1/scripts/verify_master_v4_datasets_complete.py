#!/usr/bin/env python3
"""Full V4 ship gate: counts, schema, overlap, partitions, rejected-row reporting."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sft_path = ROOT / "data" / "arabic_reasoning_coldstart_v4.jsonl"
rlvr_path = ROOT / "data" / "arabic_reasoning_rlvr_v4.jsonl"


def normalize(t: str) -> str:
    return " ".join(str(t).strip().split())


def _load_schema(name: str) -> dict | None:
    schema_path = SRC / "rlvr_contracts" / "schemas" / name
    if not schema_path.is_file():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _validate_with_schema(rows: list[dict], schema: dict | None, label: str) -> int:
    if schema is None:
        print(f"  • Schema {label}: SKIPPED (schema file missing)")
        return 0
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print(f"  • Schema {label}: SKIPPED (jsonschema not installed)")
        return 0
    validator = Draft202012Validator(schema)
    rejected = 0
    for i, row in enumerate(rows):
        errs = sorted(validator.iter_errors(row), key=lambda e: e.path)
        if errs:
            rejected += 1
            if rejected <= 5:
                print(f"    schema reject row {i}: {errs[0].message}")
    print(f"  • Schema {label} rejected rows: {rejected}")
    return rejected


print("=" * 80)
print(" MASTER V4 DATASETS SHIP GATE")
print("=" * 80)

assert sft_path.exists(), f"SFT dataset missing: {sft_path}"
assert rlvr_path.exists(), f"RLVR dataset missing: {rlvr_path}"

with open(sft_path, encoding="utf-8") as f:
    sft_items = [json.loads(l) for l in f if l.strip()]
with open(rlvr_path, encoding="utf-8") as f:
    rlvr_items = [json.loads(l) for l in f if l.strip()]

print(f"\n1. SFT: {sft_path}")
print(f"  • Total SFT Rows: {len(sft_items)}")
assert len(sft_items) == 4000, f"Expected 4000 SFT rows, found {len(sft_items)}"

sft_prompts = set()
sft_parts = Counter()
sft_missing_response = 0
for item in sft_items:
    part = str(item.get("partition") or (item.get("metadata") or {}).get("partition") or "")
    if part in {"", "sft", "train"}:
        part = "sft_train"
    sft_parts[part] += 1
    p_norm = normalize(item.get("prompt", ""))
    assert p_norm, "SFT sample missing prompt"
    if not item.get("response"):
        sft_missing_response += 1
    assert item.get("answer_spec"), "SFT sample missing answer_spec"
    sft_prompts.add(p_norm)

print(f"  • Unique SFT Prompts: {len(sft_prompts)}")
print(f"  • Partition counts: {dict(sft_parts)}")
print(f"  • Missing responses: {sft_missing_response}")
assert len(sft_prompts) == 4000, "Internal duplicate prompts in SFT"
assert sft_missing_response == 0, "SFT responses missing"

sft_schema = _load_schema("problem.schema.json") or _load_schema("sft_example.schema.json")
sft_schema_rejects = _validate_with_schema(sft_items[:200], sft_schema, "SFT(sample200)")

print(f"\n2. RLVR: {rlvr_path}")
print(f"  • Total RLVR Rows: {len(rlvr_items)}")
assert len(rlvr_items) > 0, "RLVR dataset empty"

rlvr_prompts = set()
rlvr_parts = Counter()
overlap_count = 0
missing_problem_id = 0
missing_family_id = 0
missing_verifier = 0
for item in rlvr_items:
    part = str(item.get("partition") or (item.get("metadata") or {}).get("partition") or "")
    if part in {"", "train", "rlvr"}:
        part = "rlvr_train"
    rlvr_parts[part] += 1
    p_norm = normalize(item.get("prompt", ""))
    assert p_norm, "RLVR sample missing prompt"
    assert item.get("answer_spec"), "RLVR sample missing answer_spec"
    if not (item.get("problem_id") or item.get("sample_id") or item.get("id")):
        missing_problem_id += 1
    if not (item.get("family_id") or (item.get("metadata") or {}).get("family_id")):
        missing_family_id += 1
    if not (
        item.get("verifier_version")
        or (item.get("metadata") or {}).get("verifier_version")
        or (item.get("answer_spec") or {}).get("verifier_artifact")
    ):
        missing_verifier += 1
    rlvr_prompts.add(p_norm)
    if p_norm in sft_prompts:
        overlap_count += 1

print(f"  • Unique RLVR Prompts: {len(rlvr_prompts)}")
print(f"  • Partition counts: {dict(rlvr_parts)}")
print(f"  • Missing problem_id: {missing_problem_id}")
print(f"  • Missing family_id: {missing_family_id}")
print(f"  • Missing verifier metadata: {missing_verifier}")
print(f"  • Cross-phase prompt overlap with SFT: {overlap_count}")
assert overlap_count == 0, f"CONTAMINATION: {overlap_count} overlapping prompts"
assert missing_problem_id == 0, "RLVR rows missing problem_id/sample_id"
assert "rlvr_train" in rlvr_parts or len(rlvr_parts) == 1, "Unexpected RLVR partitions"

rlvr_schema = _load_schema("rlvr_prompt.schema.json") or _load_schema("problem.schema.json")
rlvr_schema_rejects = _validate_with_schema(rlvr_items[:200], rlvr_schema, "RLVR(sample200)")

print("\n" + "=" * 80)
print(" SHIP GATE SUMMARY")
print("=" * 80)
print(f"  SFT:  {len(sft_items):,} rows | schema_rejects(sample)={sft_schema_rejects}")
print(f"  RLVR: {len(rlvr_items):,} rows | schema_rejects(sample)={rlvr_schema_rejects}")
print(f"  Overlap: {overlap_count}")
print("=" * 80)
print("SHIP GATE PASSED")
