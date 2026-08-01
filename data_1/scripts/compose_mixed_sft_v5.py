#!/usr/bin/env python3
"""Compose mixed V5 SFT: core reasoning + IFEval sidecar (no MCQ)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=Path, required=True, help="Core SFT JSONL (target 4000)")
    ap.add_argument("--ifeval", type=Path, required=True, help="IFEval sidecar JSONL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", default=True)
    args = ap.parse_args()

    core = _read_jsonl(args.core)
    ifeval = _read_jsonl(args.ifeval)
    if not core:
        raise SystemExit(f"empty core: {args.core}")
    if not ifeval:
        raise SystemExit(f"empty ifeval sidecar: {args.ifeval}")

    # Reject accidental MCQ sidecars.
    for row in ifeval:
        spec = row.get("answer_spec") or {}
        if spec.get("type") == "mcq_letter" or row.get("domain") in {
            "arapro_knowledge",
            "arapro_mcq",
        }:
            raise SystemExit("MCQ data forbidden in IFEval compose path")

    core_ids = {r.get("family_id") or r.get("problem_id") for r in core}
    overlap = [
        r
        for r in ifeval
        if (r.get("family_id") or r.get("problem_id")) in core_ids
    ]
    if overlap:
        raise SystemExit(f"family overlap between core and ifeval: {len(overlap)}")

    mixed = list(core) + list(ifeval)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(mixed)

    _write_jsonl(args.out, mixed)
    counts = Counter(str(r.get("domain")) for r in mixed)
    manifest = {
        "rows_total": len(mixed),
        "rows_core": len(core),
        "rows_ifeval": len(ifeval),
        "rows_mcq": 0,
        "domain_counts": dict(counts),
        "out": str(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
