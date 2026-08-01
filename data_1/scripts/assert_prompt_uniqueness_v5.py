#!/usr/bin/env python3
"""Fail closed on exact normalized-prompt collisions within/between SFT and RLVR."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rlvr_synth.decontam.pipeline import exact_hash, normalize_text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prompt_key(row: dict[str, Any]) -> str:
    return normalize_text(str(row.get("prompt") or ""))


def _dup_report(rows: list[dict[str, Any]], label: str) -> list[str]:
    keys = [_prompt_key(r) for r in rows]
    counts = Counter(k for k in keys if k)
    dups = {k: c for k, c in counts.items() if c > 1}
    if not dups:
        return []
    # Surface a few examples for debugging.
    examples = []
    for k, c in list(dups.items())[:5]:
        examples.append(f"{label}: count={c} hash={exact_hash(k)[:12]} prompt[:80]={k[:80]!r}")
    return [
        f"{label}: {sum(c - 1 for c in dups.values())} exact normalized-prompt duplicate(s) "
        f"across {len(dups)} unique prompt(s)"
    ] + examples


def _cross_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> list[str]:
    left_keys = {_prompt_key(r) for r in left if _prompt_key(r)}
    overlap = [r for r in right if _prompt_key(r) in left_keys]
    if not overlap:
        return []
    sample = _prompt_key(overlap[0])[:80]
    return [
        f"{left_label}↔{right_label}: {len(overlap)} exact normalized-prompt overlap(s); "
        f"example[:80]={sample!r}"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft", type=Path, action="append", default=[], help="SFT JSONL (repeatable)")
    ap.add_argument("--rlvr", type=Path, action="append", default=[], help="RLVR JSONL (repeatable)")
    ap.add_argument(
        "--drop-sft-dups",
        action="store_true",
        help="Drop later SFT exact-prompt dups and rewrite first --sft path",
    )
    ap.add_argument(
        "--drop-rlvr-dups",
        action="store_true",
        help="Drop RLVR rows overlapping SFT or within-RLVR dups; rewrite first --rlvr path",
    )
    ap.add_argument("--fail", action="store_true", default=True, help="Exit non-zero on collisions")
    ap.add_argument("--no-fail", action="store_false", dest="fail")
    args = ap.parse_args()

    if not args.sft and not args.rlvr:
        raise SystemExit("provide at least one --sft or --rlvr")

    sft_rows: list[dict[str, Any]] = []
    for p in args.sft:
        if not p.exists():
            raise SystemExit(f"missing SFT: {p}")
        sft_rows.extend(_read_jsonl(p))

    rlvr_rows: list[dict[str, Any]] = []
    for p in args.rlvr:
        if not p.exists():
            raise SystemExit(f"missing RLVR: {p}")
        rlvr_rows.extend(_read_jsonl(p))

    errors: list[str] = []
    errors.extend(_dup_report(sft_rows, "SFT"))
    errors.extend(_dup_report(rlvr_rows, "RLVR"))
    if sft_rows and rlvr_rows:
        errors.extend(_cross_overlap(sft_rows, rlvr_rows, "SFT", "RLVR"))

    if args.drop_sft_dups and args.sft:
        seen: set[str] = set()
        kept: list[dict[str, Any]] = []
        dropped = 0
        for row in sft_rows:
            k = _prompt_key(row)
            if not k or k in seen:
                dropped += 1
                continue
            seen.add(k)
            kept.append(row)
        out = args.sft[0]
        with out.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"rewrote_sft": str(out), "kept": len(kept), "dropped": dropped}))
        sft_rows = kept
        errors = [e for e in errors if not e.startswith("SFT:")]

    if args.drop_rlvr_dups and args.rlvr:
        sft_keys = {_prompt_key(r) for r in sft_rows if _prompt_key(r)}
        seen_r: set[str] = set()
        kept_r: list[dict[str, Any]] = []
        dropped_r = 0
        for row in rlvr_rows:
            k = _prompt_key(row)
            if not k or k in seen_r or k in sft_keys:
                dropped_r += 1
                continue
            seen_r.add(k)
            kept_r.append(row)
        out_r = args.rlvr[0]
        with out_r.open("w", encoding="utf-8") as f:
            for row in kept_r:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"rewrote_rlvr": str(out_r), "kept": len(kept_r), "dropped": dropped_r}))
        rlvr_rows = kept_r
        errors = [
            e
            for e in errors
            if not e.startswith("RLVR:") and "SFT↔RLVR" not in e
        ]

    report = {
        "sft_rows": len(sft_rows),
        "rlvr_rows": len(rlvr_rows),
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors and args.fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
