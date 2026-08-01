#!/usr/bin/env python3
"""Select exactly 4,000 ship-ready SFT rows from candidate corpora."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_synth.quality.acceptance import accept_sft_row, stamp_fresh_sft_audit
from rlvr_synth.quality.quotas import SFT_DOMAIN_QUOTAS, select_domain_quota


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
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Candidate SFT JSONL (e.g. release_corpora/sft_train.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--require-decontam-clean", action="store_true", default=True)
    parser.add_argument("--allow-missing-decontam", action="store_true")
    args = parser.parse_args()

    require_decontam = not args.allow_missing_decontam
    raw = _read_jsonl(args.candidates)
    accepted: list[dict] = []
    rejected: list[dict] = []

    seen_pid: set[str] = set()
    seen_fid: set[str] = set()
    seen_prompt: set[str] = set()

    for row in raw:
        stamped = stamp_fresh_sft_audit(row)
        ok, reasons = accept_sft_row(stamped, require_decontam_clean=require_decontam)
        pid = str(stamped.get("problem_id") or "")
        fid = str(stamped.get("family_id") or "")
        prompt = " ".join(str(stamped.get("prompt") or "").split())
        if not ok:
            rejected.append({**stamped, "rejection_reasons": reasons})
            continue
        if pid in seen_pid or fid in seen_fid or prompt in seen_prompt:
            rejected.append({**stamped, "rejection_reasons": ["duplicate_id_or_prompt"]})
            continue
        seen_pid.add(pid)
        seen_fid.add(fid)
        seen_prompt.add(prompt)
        accepted.append(stamped)

    selected, shortages = select_domain_quota(accepted, quotas=SFT_DOMAIN_QUOTAS)
    # Difficulty range check from programmatic num_steps
    steps = []
    for row in selected:
        ns = (row.get("metadata") or {}).get("num_steps")
        if ns is not None:
            steps.append(int(ns))
    trivial_share = (sum(1 for s in steps if s <= 1) / len(steps)) if steps else 0.0
    hard_share = (sum(1 for s in steps if s >= 4) / len(steps)) if steps else 0.0

    stats = {
        "candidates": len(raw),
        "accepted_pre_quota": len(accepted),
        "selected": len(selected),
        "rejected": len(rejected),
        "shortages": shortages,
        "domain_counts": dict(Counter(str(r.get("domain")) for r in selected)),
        "trivial_share": trivial_share,
        "hard_share": hard_share,
        "trivial_ok": trivial_share <= 0.35 + 1e-9,
        "hard_ok": hard_share >= 0.05 - 1e-9 or not steps,
    }
    _write_jsonl(args.out, selected)
    rejected_path = args.rejected or args.out.with_name(args.out.stem + ".rejected.jsonl")
    _write_jsonl(rejected_path, rejected)
    stats_path = args.stats or args.out.with_name(args.out.stem + ".STATS.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if shortages:
        print(f"ERROR: domain shortages {shortages}", file=sys.stderr)
        return 2
    if len(selected) != 4000:
        print(f"ERROR: expected 4000 selected, got {len(selected)}", file=sys.stderr)
        return 2
    if not stats["trivial_ok"] or not stats["hard_ok"]:
        print(
            f"ERROR: difficulty mix failed trivial={trivial_share:.3f} hard={hard_share:.3f}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
