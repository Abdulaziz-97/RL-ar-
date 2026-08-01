#!/usr/bin/env python3
"""Atomically promote staged V4 SFT+RLVR corpora after a full ship gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION
from rlvr_synth.release.manifest import build_manifest, write_manifest, sha256_jsonl
from rlvr_synth.release.ship_gate import run_ship_gate, write_ship_report


def _atomic_replace(src: Path, dst: Path, backup_dir: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"{dst.name}.{stamp}.bak"
        shutil.copy2(dst, backup)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rlvr", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--release-id", type=str, required=True)
    parser.add_argument(
        "--sft-dest",
        type=Path,
        default=ROOT / "data" / "arabic_reasoning_coldstart_v4.jsonl",
    )
    parser.add_argument(
        "--rlvr-dest",
        type=Path,
        default=ROOT / "data" / "arabic_reasoning_rlvr_v4.jsonl",
    )
    parser.add_argument("--skip-promote", action="store_true",
                        help="Run ship gate only; do not overwrite production")
    parser.add_argument("--require-human-review-report", type=Path, default=None)
    args = parser.parse_args()

    staging = args.staging_root / args.release_id
    staging.mkdir(parents=True, exist_ok=True)

    if args.require_human_review_report is not None:
        report = json.loads(args.require_human_review_report.read_text(encoding="utf-8"))
        if not report.get("pass"):
            print(json.dumps({"error": "human_review_failed", "report": report}, indent=2))
            return 2

    corpora = {"sft_train": args.sft, "rlvr_train": args.rlvr}
    schema_kinds = {"sft_train": "sft_trace", "rlvr_train": "rlvr_prompt"}
    artifacts = [
        ("sft_train", "arabic_reasoning_coldstart_v4.jsonl", args.sft),
        ("rlvr_train", "arabic_reasoning_rlvr_v4.jsonl", args.rlvr),
    ]
    manifest = build_manifest(
        args.release_id,
        corpus_version=args.release_id,
        verifier_registry_version=VERIFIER_REGISTRY_VERSION,
        artifacts=artifacts,
        notes={
            "sft_sha256": sha256_jsonl(args.sft),
            "rlvr_sha256": sha256_jsonl(args.rlvr),
            "pipeline": "v4_dual_corpus_promote",
        },
    )
    man_path = write_manifest(manifest, staging)
    gate = run_ship_gate(
        corpora=corpora,
        manifest=manifest,
        root=staging,
        schema_kind_by_corpus=schema_kinds,
        require_exact_counts={"sft_train": 4000, "rlvr_train": 4000},
        require_domain_quotas=True,
        require_rlvr_band_matrix=True,
        require_per_row_arabic=True,
        require_decontam_clean=True,
        require_leak_clean=True,
    )
    write_ship_report(gate, staging / "ship_gate_report.json")
    summary = {
        "manifest": str(man_path),
        "passed": gate.passed,
        "errors": gate.errors,
        "promoted": False,
    }
    if not gate.passed:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    if args.skip_promote:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    backup_dir = staging / "backups"
    _atomic_replace(args.sft, args.sft_dest, backup_dir)
    _atomic_replace(args.rlvr, args.rlvr_dest, backup_dir)
    # Copy immutable manifest next to production data.
    prod_manifest = ROOT / "data" / "release_manifest_v4.json"
    shutil.copy2(man_path, prod_manifest)
    summary["promoted"] = True
    summary["sft_dest"] = str(args.sft_dest)
    summary["rlvr_dest"] = str(args.rlvr_dest)
    summary["prod_manifest"] = str(prod_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
