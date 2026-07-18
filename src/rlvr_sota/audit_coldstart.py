"""
Audit cold-start CoT JSONL for tag-boundary purity (structural leak precursors).

Flags responses with:
  - missing / empty <think> or <answer> tags
  - preamble/postamble text outside tags (penalty_structural_leak)
  - low metadata quality / arabic_purity when present

Usage:
  python -m rlvr_sota.audit_coldstart --data data/arabic_reasoning_coldstart.jsonl
  python scripts/audit_coldstart.py --data data/arabic_reasoning_coldstart.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rlvr.reward_composer import (
    penalty_structural_leak,
    reward_format,
)
from rlvr_sota.data import _transform_coldstart_response


@dataclass
class ColdstartIssue:
    sample_id: str
    line_no: int
    reasons: list[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    arabic_purity: Optional[float] = None
    preview: str = ""


@dataclass
class AuditReport:
    total: int = 0
    skipped_empty_response: int = 0
    clean: int = 0
    issues: list[ColdstartIssue] = field(default_factory=list)
    reason_counts: Counter = field(default_factory=Counter)

    @property
    def impure_count(self) -> int:
        return len(self.issues)

    @property
    def impure_rate(self) -> float:
        return self.impure_count / self.total if self.total else 0.0


def _preview(text: str, n: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 3] + "..."


def audit_coldstart_record(
    raw: dict[str, Any],
    *,
    line_no: int,
    max_preamble_words: int = 5,
    min_quality: float = 0.5,
    min_arabic_purity: float = 0.7,
) -> Optional[ColdstartIssue]:
    response = raw.get("response", "") or ""
    if not response.strip():
        return None

    transformed = _transform_coldstart_response(response)
    reasons: list[str] = []

    if "<think>" not in transformed or "</think>" not in transformed:
        reasons.append("missing_think_tags")
    if "<answer>" not in transformed or "</answer>" not in transformed:
        reasons.append("missing_answer_tags")

    fmt = reward_format(transformed)
    if fmt <= 0.0 and "missing_think_tags" not in reasons and "missing_answer_tags" not in reasons:
        reasons.append("format_reward_zero")

    if penalty_structural_leak(transformed, max_preamble_words=max_preamble_words) > 0.0:
        reasons.append("structural_leak_outside_tags")

    quality = raw.get("quality") or {}
    q_score = quality.get("score")
    purity = quality.get("arabic_purity")
    if isinstance(q_score, (int, float)) and q_score < min_quality:
        reasons.append(f"low_quality_score<{min_quality}")
    if isinstance(purity, (int, float)) and purity < min_arabic_purity:
        reasons.append(f"low_arabic_purity<{min_arabic_purity}")

    if not reasons:
        return None

    return ColdstartIssue(
        sample_id=str(raw.get("id", "")),
        line_no=line_no,
        reasons=reasons,
        quality_score=float(q_score) if isinstance(q_score, (int, float)) else None,
        arabic_purity=float(purity) if isinstance(purity, (int, float)) else None,
        preview=_preview(transformed),
    )


def audit_coldstart_file(
    path: str | Path,
    *,
    max_preamble_words: int = 5,
    min_quality: float = 0.5,
    min_arabic_purity: float = 0.7,
    limit: Optional[int] = None,
) -> AuditReport:
    report = AuditReport()
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and report.total >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            report.total += 1

            if not (raw.get("response") or "").strip():
                report.skipped_empty_response += 1
                continue

            issue = audit_coldstart_record(
                raw,
                line_no=line_no,
                max_preamble_words=max_preamble_words,
                min_quality=min_quality,
                min_arabic_purity=min_arabic_purity,
            )
            if issue is None:
                report.clean += 1
            else:
                report.issues.append(issue)
                for reason in issue.reasons:
                    # Normalize parameterized reasons for counting.
                    key = reason.split("<")[0] if "<" in reason else reason
                    report.reason_counts[key] += 1
    return report


def format_report(report: AuditReport, *, top_n: int = 20) -> str:
    lines = [
        "Cold-start purity audit",
        f"  total records:     {report.total}",
        f"  empty responses:   {report.skipped_empty_response}",
        f"  clean:             {report.clean}",
        f"  impure:            {report.impure_count} ({100 * report.impure_rate:.1f}%)",
        "  reason breakdown:",
    ]
    for reason, count in report.reason_counts.most_common():
        lines.append(f"    - {reason}: {count}")

    if report.issues:
        lines.append(f"  top {min(top_n, len(report.issues))} impure samples:")
        for issue in report.issues[:top_n]:
            lines.append(
                f"    line {issue.line_no} id={issue.sample_id!r} "
                f"reasons={issue.reasons} q={issue.quality_score} "
                f"purity={issue.arabic_purity}"
            )
            lines.append(f"      preview: {issue.preview}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cold-start CoT tag purity")
    parser.add_argument(
        "--data",
        default="data/arabic_reasoning_coldstart.jsonl",
        help="Path to cold-start JSONL",
    )
    parser.add_argument("--max-preamble-words", type=int, default=5)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--min-arabic-purity", type=float, default=0.7)
    parser.add_argument("--top", type=int, default=20, help="How many impure samples to print")
    parser.add_argument("--json-out", help="Optional path to write full JSON report")
    parser.add_argument("--fail-above", type=float, default=None,
                        help="Exit 1 if impure_rate exceeds this fraction (e.g. 0.1)")
    args = parser.parse_args(argv)

    report = audit_coldstart_file(
        args.data,
        max_preamble_words=args.max_preamble_words,
        min_quality=args.min_quality,
        min_arabic_purity=args.min_arabic_purity,
    )
    text = format_report(report, top_n=args.top)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))

    if args.json_out:
        payload = {
            "total": report.total,
            "skipped_empty_response": report.skipped_empty_response,
            "clean": report.clean,
            "impure": report.impure_count,
            "impure_rate": report.impure_rate,
            "reason_counts": dict(report.reason_counts),
            "issues": [
                {
                    "sample_id": i.sample_id,
                    "line_no": i.line_no,
                    "reasons": i.reasons,
                    "quality_score": i.quality_score,
                    "arabic_purity": i.arabic_purity,
                    "preview": i.preview,
                }
                for i in report.issues
            ],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote JSON report to {args.json_out}")

    if args.fail_above is not None and report.impure_rate > args.fail_above:
        print(
            f"\nFAIL: impure_rate {report.impure_rate:.3f} > threshold {args.fail_above}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
