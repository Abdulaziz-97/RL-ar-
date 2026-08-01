#!/usr/bin/env python3
"""Full V4 dual-corpus ship gate: exact counts, schema, quotas, overlap, manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT / "src"), str(ROOT / "src"), str(PACK_ROOT)]

from rlvr_synth.release.manifest import ReleaseManifest
from rlvr_synth.release.ship_gate import run_ship_gate, write_ship_report

sft_path = ROOT / "data" / "arabic_reasoning_coldstart_v4.jsonl"
rlvr_path = ROOT / "data" / "arabic_reasoning_rlvr_v4.jsonl"
manifest_path = ROOT / "data" / "release_manifest_v4.json"


def main() -> int:
    print("=" * 80)
    print(" MASTER V4 DATASETS SHIP GATE (full-corpus, fail-closed)")
    print("=" * 80)

    assert sft_path.exists(), f"SFT dataset missing: {sft_path}"
    assert rlvr_path.exists(), f"RLVR dataset missing: {rlvr_path}"

    corpora = {"sft_train": sft_path, "rlvr_train": rlvr_path}
    schema_kinds = {"sft_train": "sft_trace", "rlvr_train": "rlvr_prompt"}

    manifest = None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Minimal reconstruct for registry/stale checks when present.
        from rlvr_synth.release.manifest import ArtifactRef

        manifest = ReleaseManifest(
            release_id=payload.get("release_id", "v4"),
            created_at=payload.get("created_at", ""),
            corpus_version=payload.get("corpus_version", "v4"),
            verifier_registry_version=payload.get("verifier_registry_version", ""),
            artifacts=[ArtifactRef(**a) for a in payload.get("artifacts", [])],
            qa_report_sha256=payload.get("qa_report_sha256", ""),
            candidate_archive_sha256=payload.get("candidate_archive_sha256", ""),
            quarantine_archive_sha256=payload.get("quarantine_archive_sha256", ""),
            notes=payload.get("notes") or {},
            immutable=bool(payload.get("immutable", True)),
        )

    report = run_ship_gate(
        corpora=corpora,
        manifest=manifest,
        root=ROOT / "data",
        schema_kind_by_corpus=schema_kinds,
        require_exact_counts={"sft_train": 4000, "rlvr_train": 4000},
        require_domain_quotas=True,
        require_rlvr_band_matrix=True,
        require_per_row_arabic=True,
        require_decontam_clean=True,
        require_leak_clean=True,
    )
    out = ROOT / "data" / "ship_gate_report_v4.json"
    write_ship_report(report, out)

    print(json.dumps({"passed": report.passed, "errors": report.errors, "report": str(out)}, ensure_ascii=False, indent=2))
    print("=" * 80)
    if report.passed:
        print("SHIP GATE PASSED")
        return 0
    print("SHIP GATE FAILED")
    for err in report.errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
