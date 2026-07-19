"""Check that replay output matches the dial-20 reference corpus.

Compares line-normalized SHA-256 of `outputs/run/release_corpora/sft_train.jsonl`
against `reference_output/sft_train.jsonl`. Pass criterion is **hash match**.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]


def sha256_jsonl(path: Path) -> str:
    """SHA-256 over non-empty stripped JSONL lines (stable across trailing blanks)."""
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def load_ids(path: Path) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["problem_id"])
    return ids


def main() -> int:
    ref = PACK_ROOT / "reference_output" / "sft_train.jsonl"
    got = PACK_ROOT / "outputs" / "run" / "release_corpora" / "sft_train.jsonl"
    if not got.exists():
        print(f"MISSING output: {got}")
        return 1
    if not ref.exists():
        print(f"MISSING reference: {ref}")
        return 1

    href, hgot = sha256_jsonl(ref), sha256_jsonl(got)
    ref_ids, got_ids = load_ids(ref), load_ids(got)
    same_hash = href == hgot
    report = {
        "ref_sha256": href,
        "got_sha256": hgot,
        "hash_match": same_hash,
        "id_order_match": ref_ids == got_ids,
        "id_set_match": set(ref_ids) == set(got_ids),
        "ref_n": len(ref_ids),
        "got_n": len(got_ids),
        "pass": same_hash,
    }
    print(json.dumps(report, indent=2))
    (PACK_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (PACK_ROOT / "outputs" / "verify_repro.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
