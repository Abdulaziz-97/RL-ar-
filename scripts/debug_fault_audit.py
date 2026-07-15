"""
Debug fault audit for RLVR pipeline — session 16ba01.
Exercises remaining-fault hypotheses without requiring GPU/TRL training.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "debug-16ba01.log"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "16ba01",
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[{hypothesis_id}] {message}: {data}")


def _serialize_ground_truth(gt):
    if gt is None:
        return ""
    if isinstance(gt, bool):
        return "true" if gt else "false"
    if isinstance(gt, (int, float)):
        return str(gt)
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False)
    return str(gt)


def audit_hypothesis_a_crps_format_gate() -> None:
    from rlvr.reward_composer import reward_format, reward_correctness

    cases = [
        ("short_correct", "<think>خطوة واحد خطوتان ثلاث</think><answer>42</answer>", 42),
        ("long_correct", "<think>" + " ".join([f"خطوة{i}" for i in range(12)]) + "</think><answer>42</answer>", 42),
        ("repetitive", "<think>" + " ".join(["نفس"] * 20) + "</think><answer>42</answer>", 42),
        ("bare_answer", "<answer>42</answer>", 42),
    ]
    for name, completion, gt in cases:
        fmt = reward_format(completion)
        corr = reward_correctness(completion, gt, "math")
        would_record = corr >= 1.0 - 1e-6 and fmt > 0.0
        _log(
            "A",
            "debug_fault_audit.py:A",
            "CRPS gate on synthetic completion",
            {
                "case": name,
                "correctness": corr,
                "format": fmt,
                "would_record_crps_trace": would_record,
                "runId": "post-fix",
            },
        )


def audit_hypothesis_b_dataset_filter() -> None:
    data_path = ROOT / "data" / "arabic_reasoning_rlvr.jsonl"
    counts = {"total": 0, "mmlu": 0, "missing_gta": 0, "kept_estimate": 0, "logic": 0}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["total"] += 1
            domain = row.get("domain")
            md = row.get("metadata") or {}
            gt = md.get("ground_truth_answer")
            if domain == "mmlu":
                counts["mmlu"] += 1
                continue
            if gt is None or gt == "":
                counts["missing_gta"] += 1
                continue
            counts["kept_estimate"] += 1
            if domain == "logic":
                counts["logic"] += 1

    logic_ser_types = set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("domain") != "logic":
                continue
            gt = (row.get("metadata") or {}).get("ground_truth_answer")
            if gt is None:
                continue
            ser = _serialize_ground_truth(gt)
            logic_ser_types.add(type(ser).__name__)

    _log(
        "B",
        "debug_fault_audit.py:B",
        "dataset filter estimate",
        {
            **counts,
            "logic_serialized_types": sorted(logic_ser_types),
            "mmlu_excluded_by_domain_skip": True,
        },
    )


def audit_hypothesis_c_yaml_weights() -> None:
    yaml_path = ROOT / "configs" / "qwen_4b_qlora.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    m = re.search(r"reward_weights:\s*\[([^\]]+)\]", text)
    yaml_weights = [float(x.strip()) for x in m.group(1).split(",")] if m else None
    m2 = re.search(r"max_completion_length:\s*(\d+)", text)
    yaml_maxlen = int(m2.group(1)) if m2 else None

    from rlvr.reward_composer import (
        W_LANGUAGE,
        W_CORRECTNESS,
        W_FORMAT,
        W_ANSWER_LEAK,
        W_STRUCTURAL_LEAK,
    )

    default_weights = [W_CORRECTNESS, W_FORMAT, W_LANGUAGE, W_ANSWER_LEAK, W_STRUCTURAL_LEAK]

    # Simulate build_trainer weight resolution after BUG-01 fix
    reward_weights = yaml_weights if yaml_weights else default_weights

    _log(
        "C",
        "debug_fault_audit.py:C",
        "YAML vs code reward weight resolution",
        {
            "yaml_reward_weights": yaml_weights,
            "yaml_max_completion_length": yaml_maxlen,
            "code_W_LANGUAGE": W_LANGUAGE,
            "DEFAULT_REWARD_WEIGHTS": list(default_weights),
            "effective_after_build_trainer": list(reward_weights),
            "stale_lang_weight_applied": abs(float(reward_weights[2]) - 0.2) < 1e-9,
            "intended_lang_weight_0_05": abs(W_LANGUAGE - 0.05) < 1e-9,
        },
    )


def audit_hypothesis_d_logic_gt_mismatch() -> None:
    from rlvr.reward_composer import reward_correctness

    data_path = ROOT / "data" / "arabic_reasoning_rlvr.jsonl"
    logic_rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("domain") != "logic":
                continue
            gt = (row.get("metadata") or {}).get("ground_truth_answer")
            if gt is None:
                continue
            logic_rows.append((row.get("id"), gt))
            if len(logic_rows) >= 5:
                break

    mismatches = 0
    for sample_id, gt in logic_rows:
        serialized = _serialize_ground_truth(gt)
        answer_json = json.dumps(gt, ensure_ascii=False)
        completion = f"<think>{' '.join(['تحليل'] * 12)}</think><answer>{answer_json}</answer>"
        score_with_serialized = reward_correctness(completion, serialized, "logic")
        score_with_raw_dict = reward_correctness(completion, gt, "logic")
        bug = score_with_serialized < 0.5 and score_with_raw_dict >= 0.5
        if bug:
            mismatches += 1
        _log(
            "D",
            "debug_fault_audit.py:D",
            "logic GT serialize-then-score",
            {
                "sample_id": sample_id,
                "raw_gt_type": type(gt).__name__,
                "serialized_type": type(serialized).__name__,
                "score_with_serialized_gt": score_with_serialized,
                "score_with_raw_dict_gt": score_with_raw_dict,
                "bug_reproduced": bug,
            },
        )

    _log(
        "D",
        "debug_fault_audit.py:D_summary",
        "logic mismatch summary",
        {"n_tested": len(logic_rows), "n_mismatched": mismatches},
    )


def audit_hypothesis_e_crps_key_and_counter() -> None:
    from rlvr.crps import get_crps_suffix_fraction

    counter = {}
    fractions = []
    key = "rlvr_logic_0042:logic:hard"
    for _ in range(5):
        failure_count = counter.get(key, 0)
        frac = get_crps_suffix_fraction(failure_count)
        fractions.append(frac)
        counter[key] = failure_count + 1

    _log(
        "E",
        "debug_fault_audit.py:E",
        "CRPS counter schedule simulation",
        {
            "per_sample_fractions": fractions,
            "expected": [0.0, 0.1, 0.3, 0.5, 0.5],
            "schedule_ok": fractions == [0.0, 0.1, 0.3, 0.5, 0.5],
        },
    )


def main() -> int:
    print(f"Writing logs to {LOG}")
    audit_hypothesis_a_crps_format_gate()
    audit_hypothesis_b_dataset_filter()
    audit_hypothesis_c_yaml_weights()
    audit_hypothesis_d_logic_gt_mismatch()
    audit_hypothesis_e_crps_key_and_counter()
    print("Audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
