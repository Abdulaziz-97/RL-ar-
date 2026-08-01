#!/usr/bin/env python3
"""Export stratified human-review packs and optionally summarize annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src")]

from rlvr_synth.qa.human_review import (
    export_review_pack,
    import_review_pack,
    stratified_sample,
    summarize_reviews,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--corpus", type=Path, required=True)
    p_export.add_argument("--out", type=Path, required=True)
    p_export.add_argument("--n", type=int, default=100)
    p_export.add_argument("--seed", type=int, default=0)
    p_export.add_argument(
        "--strata",
        type=str,
        default="domain,template_family",
        help="Comma-separated stratum keys; template_family resolved from metadata/lineage",
    )
    p_export.add_argument("--kind", choices=["sft", "rlvr"], default="sft")

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--pack", type=Path, required=True)
    p_sum.add_argument("--out", type=Path, required=True)
    p_sum.add_argument("--min-accept-rate", type=float, default=0.95)
    p_sum.add_argument("--min-kappa", type=float, default=0.6)
    p_sum.add_argument("--min-wilson-low", type=float, default=0.9)

    args = parser.parse_args()
    if args.cmd == "export":
        rows = _read_jsonl(args.corpus)
        # Materialize template_family for stratification.
        for row in rows:
            tf = (
                (row.get("lineage") or {}).get("template_family")
                or (row.get("metadata") or {}).get("template_family")
                or row.get("domain")
            )
            row["template_family"] = tf
            if args.kind == "rlvr":
                row["empirical_band"] = (
                    (row.get("empirical_difficulty") or {}).get("release_band")
                    or (row.get("empirical_difficulty") or {}).get("band")
                    or (row.get("metadata") or {}).get("difficulty_band")
                    or ""
                )
        keys = tuple(k.strip() for k in args.strata.split(",") if k.strip())
        if args.kind == "rlvr" and "empirical_band" not in keys:
            keys = keys + ("empirical_band",)
        items = stratified_sample(rows, n=args.n, strata_keys=keys, seed=args.seed)
        export_review_pack(items, args.out)
        print(json.dumps({"n": len(items), "out": str(args.out)}, indent=2))
        return 0

    pack = import_review_pack(args.pack)
    summary = summarize_reviews(
        pack,
        min_accept_rate=args.min_accept_rate,
        min_kappa=args.min_kappa,
        min_wilson_low=args.min_wilson_low,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
