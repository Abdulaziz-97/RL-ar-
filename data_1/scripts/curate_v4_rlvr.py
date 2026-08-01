#!/usr/bin/env python3
"""Curate the exact 4,000-row V4 RLVR domain×band release matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_synth.quality.acceptance import accept_rlvr_row
from rlvr_synth.quality.quotas import RLVR_BAND_MATRIX, select_rlvr_band_matrix


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True,
                        help="One or more calibrated candidate JSONL shards")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mastered", type=Path, default=None)
    parser.add_argument("--deferred", type=Path, default=None)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--allow-missing-decontam", action="store_true")
    args = parser.parse_args()

    raw: list[dict] = []
    for path in args.candidates:
        if not path.is_file():
            print(f"ERROR: candidate shard missing: {path}", file=sys.stderr)
            return 2
        raw.extend(_read_jsonl(path))
    if not raw:
        print("ERROR: no candidate rows loaded from shards", file=sys.stderr)
        return 2

    accepted = []
    rejected = []
    seen_pid: set[str] = set()
    seen_fid: set[str] = set()
    for row in raw:
        # pass@8 historically stamped top-level difficulty_tag; schema forbids it.
        if "difficulty_tag" in row:
            row = dict(row)
            tag = row.pop("difficulty_tag")
            md = dict(row.get("metadata") or {})
            md.setdefault("difficulty_tag", tag)
            row["metadata"] = md
        ok, reasons = accept_rlvr_row(
            row, require_decontam_clean=not args.allow_missing_decontam
        )
        pid = str(row.get("problem_id") or "")
        fid = str(row.get("family_id") or "")
        if pid in seen_pid or fid in seen_fid:
            rejected.append({**row, "rejection_reasons": ["duplicate_id"]})
            continue
        if not ok:
            # Allow pending_pass8 rows that only fail missing empirical fields? No —
            # accept_rlvr_row does not require bands. Keep gate failures out.
            rejected.append({**row, "rejection_reasons": reasons})
            continue
        if not (row.get("empirical_difficulty") or {}).get("band"):
            rejected.append({**row, "rejection_reasons": ["missing_empirical_band"]})
            continue
        seen_pid.add(pid)
        seen_fid.add(fid)
        accepted.append(row)

    selected, mastered, deferred, shortages = select_rlvr_band_matrix(accepted)
    stats = {
        "candidates": len(raw),
        "accepted_pre_matrix": len(accepted),
        "selected": len(selected),
        "mastered": len(mastered),
        "deferred": len(deferred),
        "rejected": len(rejected),
        "shortages": shortages,
        "domain_counts": dict(Counter(str(r.get("domain")) for r in selected)),
        "band_counts": dict(
            Counter(
                str((r.get("empirical_difficulty") or {}).get("release_band") or "")
                for r in selected
            )
        ),
        "matrix": RLVR_BAND_MATRIX,
    }

    _write_jsonl(args.out, selected)
    mastered_path = args.mastered or args.out.with_name(args.out.stem + ".mastered.jsonl")
    deferred_path = args.deferred or args.out.with_name(args.out.stem + ".deferred.jsonl")
    rejected_path = args.rejected or args.out.with_name(args.out.stem + ".rejected.jsonl")
    _write_jsonl(mastered_path, mastered)
    _write_jsonl(deferred_path, deferred)
    _write_jsonl(rejected_path, rejected)
    stats_path = args.stats or args.out.with_name(args.out.stem + ".STATS.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if shortages:
        print(f"ERROR: band/domain shortages {shortages}", file=sys.stderr)
        print(
            "Generate and calibrate a top-up tranche; do not weaken thresholds.",
            file=sys.stderr,
        )
        return 2
    if len(selected) != 4000:
        print(f"ERROR: expected 4000 selected, got {len(selected)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
