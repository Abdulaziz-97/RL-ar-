"""Tests for cold-start purity audit."""

import json

from rlvr_sota.audit_coldstart import (
    audit_coldstart_file,
    audit_coldstart_record,
    format_report,
)


def test_audit_clean_record():
    raw = {
        "id": "ok-1",
        "response": (
            "<think>أولا أجمع العددين معا ثم أحسب الناتج بدقة عالية جدا</think>"
            "<answer>4</answer>"
        ),
        "quality": {"score": 0.9, "arabic_purity": 0.95},
    }
    assert audit_coldstart_record(raw, line_no=1) is None


def test_audit_flags_structural_leak():
    raw = {
        "id": "bad-1",
        "response": (
            "First I explain outside the tags carefully here. "
            "<think>reasoning words enough here now yes</think><answer>4</answer>"
        ),
    }
    issue = audit_coldstart_record(raw, line_no=2)
    assert issue is not None
    assert "structural_leak_outside_tags" in issue.reasons


def test_audit_flags_missing_tags():
    raw = {"id": "bad-2", "response": "just plain text with #### 42"}
    issue = audit_coldstart_record(raw, line_no=3)
    assert issue is not None
    # #### transform adds <answer> but still no <think>
    assert "missing_think_tags" in issue.reasons


def test_audit_file(tmp_path):
    path = tmp_path / "cs.jsonl"
    records = [
        {
            "id": "1",
            "response": (
                "<think>أولا أجمع العددين معا ثم أحسب الناتج بدقة عالية جدا</think>"
                "<answer>4</answer>"
            ),
            "quality": {"score": 0.9, "arabic_purity": 0.95},
        },
        {
            "id": "2",
            "response": (
                "lots of preamble words before the tags here now "
                "<think>x</think><answer>1</answer>"
            ),
        },
        {"id": "3", "response": ""},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    report = audit_coldstart_file(path)
    assert report.total == 3
    assert report.skipped_empty_response == 1
    assert report.clean == 1
    assert report.impure_count == 1
    text = format_report(report)
    assert "impure" in text.lower() or "impure:" in text
