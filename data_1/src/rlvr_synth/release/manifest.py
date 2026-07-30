"""Immutable release manifests with hashes/checksums and handoff docs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonl(path: Path) -> str:
    """Content hash that normalizes line endings and ignores blank lines."""
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


@dataclass
class ArtifactRef:
    path: str
    role: str
    sha256: str
    n_rows: int | None = None


@dataclass
class ReleaseManifest:
    release_id: str
    created_at: str
    corpus_version: str
    verifier_registry_version: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    qa_report_sha256: str = ""
    candidate_archive_sha256: str = ""
    quarantine_archive_sha256: str = ""
    notes: dict[str, Any] = field(default_factory=dict)
    immutable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return sha256_bytes(payload.encode("utf-8"))


def count_jsonl_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_manifest(
    release_id: str,
    *,
    corpus_version: str,
    verifier_registry_version: str,
    artifacts: Iterable[tuple[str, str, Path]],
    qa_report: Path | None = None,
    candidate_archive: Path | None = None,
    quarantine_archive: Path | None = None,
    notes: dict[str, Any] | None = None,
) -> ReleaseManifest:
    refs = [
        ArtifactRef(
            path=relpath,
            role=role,
            sha256=sha256_jsonl(path) if path.suffix == ".jsonl" else sha256_file(path),
            n_rows=count_jsonl_rows(path) if path.suffix == ".jsonl" else None,
        )
        for role, relpath, path in artifacts
    ]
    return ReleaseManifest(
        release_id=release_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_version=corpus_version,
        verifier_registry_version=verifier_registry_version,
        artifacts=refs,
        qa_report_sha256=sha256_file(qa_report) if qa_report and qa_report.exists() else "",
        candidate_archive_sha256=(
            sha256_file(candidate_archive)
            if candidate_archive and candidate_archive.exists()
            else ""
        ),
        quarantine_archive_sha256=(
            sha256_file(quarantine_archive)
            if quarantine_archive and quarantine_archive.exists()
            else ""
        ),
        notes=notes or {},
    )


def write_manifest(manifest: ReleaseManifest, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "release_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("immutable", True):
            raise RuntimeError(
                f"refusing to overwrite immutable release manifest at {path}; "
                "create a new release_id"
            )
    payload = manifest.to_dict()
    payload["manifest_sha256"] = manifest.content_hash()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "HANDOFF.md").write_text(
        _handoff_markdown(manifest, payload["manifest_sha256"]), encoding="utf-8"
    )
    return path


def _handoff_markdown(manifest: ReleaseManifest, manifest_sha: str) -> str:
    lines = [
        f"# Release handoff: {manifest.release_id}",
        "",
        f"- created_at: `{manifest.created_at}`",
        f"- corpus_version: `{manifest.corpus_version}`",
        f"- verifier_registry_version: `{manifest.verifier_registry_version}`",
        f"- manifest_sha256: `{manifest_sha}`",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in manifest.artifacts:
        rows = f" ({artifact.n_rows} rows)" if artifact.n_rows is not None else ""
        lines.append(
            f"- `{artifact.role}` → `{artifact.path}` "
            f"sha256=`{artifact.sha256}`{rows}"
        )
    lines.extend(
        [
            "",
            "## Release notes",
            "",
            "- This package is code+manifest only unless corpus files are attached.",
            "- Do not claim private-eval / human / GPU gates passed unless reports are present.",
            "- Symbolic answer types are rejected in v1.",
            "",
        ]
    )
    return "\n".join(lines)


def detect_stale_artifacts(manifest: ReleaseManifest, root: Path) -> list[str]:
    """Return paths whose on-disk hash no longer matches the manifest."""
    stale: list[str] = []
    for artifact in manifest.artifacts:
        path = root / artifact.path
        if not path.exists():
            stale.append(f"missing:{artifact.path}")
            continue
        current = sha256_jsonl(path) if path.suffix == ".jsonl" else sha256_file(path)
        if current != artifact.sha256:
            stale.append(f"hash_mismatch:{artifact.path}")
    return stale
