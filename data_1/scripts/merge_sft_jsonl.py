#!/usr/bin/env python3
"""Merge SFT JSONL corpora without BOM / PowerShell corruption."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("inputs", nargs="+", type=Path)
    args = ap.parse_args()
    rows = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "merged": len(rows),
                "domains": dict(Counter(str(r.get("domain")) for r in rows)),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
