from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from rlvr_contracts.answer_spec import AnswerSpecError, parse_answer_spec, reject_symbolic
from rlvr_contracts.leak import leak_policy_score, structural_leak_score
from rlvr_contracts.response import parse_response
from rlvr_contracts.schema_validate import validate_record
from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION, verify_answer
from rlvr_synth.qa.arabic_metrics import score_arabic_text
from rlvr_synth.release.manifest import ReleaseManifest, detect_stale_artifacts, sha256_file

_ARABIC_RE = re.compile("[\\u0600-\\u06FF]")

try:
    from rlvr_synth.quality.quotas import CORE_DOMAINS, DOMAIN_QUOTAS, RLVR_BAND_MATRIX
except Exception:  # pragma: no cover
    CORE_DOMAINS = ("gsm8k", "math", "math_comp", "logic")
    DOMAIN_QUOTAS = {"gsm8k": 1200, "math": 1000, "math_comp": 1000, "logic": 800}
    RLVR_BAND_MATRIX = {}


@dataclass
class GateResult:
    name: str
    passed: bool
    hard_failure: bool
    detail: str
    metric: float | int | None = None


@dataclass
class ShipGateReport:
    passed: bool
    gates: list[GateResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _arabic_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha() or _ARABIC_RE.search(c)]
    if not chars:
        return 1.0
    ar = sum(1 for c in chars if _ARABIC_RE.search(c))
    return ar / len(chars)


def _infer_schema_kind(row: dict[str, Any], default: str) -> str:
    partition = str(row.get("partition", ""))
    if "access_control" in row or partition in {"private_eval", "contaminated_diagnostic"}:
        if "response" in row and row.get("response"):
            return "sft_trace"
        if partition == "contaminated_diagnostic" and (not row.get("response")):
            return "rlvr_prompt"
        return "evaluation"
    if row.get("response"):
        return "sft_trace"
    if default:
        return default
    return "rlvr_prompt"


def _decontam_status(row: dict[str, Any]) -> str:
    return str(
        (row.get("decontam") or {}).get("status")
        or ((row.get("metadata") or {}).get("decontam") or {}).get("status")
        or ""
    )


def _template_family(row: dict[str, Any]) -> str:
    return str(
        (row.get("lineage") or {}).get("template_family")
        or (row.get("metadata") or {}).get("template_family")
        or row.get("domain")
        or "unknown"
    )


def _release_band(row: dict[str, Any]) -> str:
    emp = row.get("empirical_difficulty") or {}
    meta = row.get("metadata") or {}
    return str(
        emp.get("release_band")
        or emp.get("band")
        or meta.get("difficulty_band")
        or row.get("difficulty_tag")
        or ""
    )


def audit_family_splits(corpora: dict[str, list[dict[str, Any]]]) -> list[GateResult]:
    results: list[GateResult] = []
    family_sets = {
        name: {str(r.get("family_id", "")) for r in rows if r.get("family_id")}
        for name, rows in corpora.items()
    }
    prompt_sets = {
        name: {str(r.get("prompt", "")).strip() for r in rows if r.get("prompt")}
        for name, rows in corpora.items()
    }
    pairs = list(combinations(sorted(corpora), 2))
    for a, b in pairs:
        if a not in family_sets or b not in family_sets:
            continue
        fam_overlap = family_sets[a] & family_sets[b]
        fam_overlap.discard("")
        results.append(
            GateResult(
                name=f"family_overlap:{a}|{b}",
                passed=len(fam_overlap) == 0,
                hard_failure=True,
                detail=f"overlap={len(fam_overlap)}",
                metric=len(fam_overlap),
            )
        )
        prompt_overlap = prompt_sets[a] & prompt_sets[b]
        prompt_overlap.discard("")
        results.append(
            GateResult(
                name=f"prompt_overlap:{a}|{b}",
                passed=len(prompt_overlap) == 0,
                hard_failure=True,
                detail=f"overlap={len(prompt_overlap)}",
                metric=len(prompt_overlap),
            )
        )
    return results


def run_ship_gate(
    *,
    corpora: dict[str, Path],
    manifest: ReleaseManifest | None = None,
    root: Path | None = None,
    schema_kind_by_corpus: dict[str, str] | None = None,
    min_arabic_median: float = 0.9,
    min_arabic_p5: float = 0.8,
    token_soft: int = 256,
    token_hard: int = 640,
    max_template_share: float = 0.35,
    max_template_family_share: float = 0.15,
    require_complete_metadata: bool = True,
    require_exact_counts: dict[str, int] | None = None,
    require_domain_quotas: bool = False,
    require_rlvr_band_matrix: bool = False,
    require_per_row_arabic: bool = False,
    require_decontam_clean: bool = False,
    require_leak_clean: bool = False,
) -> ShipGateReport:
    gates: list[GateResult] = []
    errors: list[str] = []
    loaded: dict[str, list[dict[str, Any]]] = {}
    schema_map = schema_kind_by_corpus or {}
    exact_counts = require_exact_counts or {}

    for name, path in corpora.items():
        rows = _read_jsonl(path)
        loaded[name] = rows
        schema_errors = 0
        answer_errors = 0
        format_errors = 0
        verifier_errors = 0
        symbolic_errors = 0
        meta_errors = 0
        leak_errors = 0
        per_row_arabic_errors = 0
        missing_decontam = 0
        non_clean_decontam = 0
        arabic_scores: list[float] = []
        token_lens: list[int] = []
        templates: Counter[str] = Counter()
        template_families: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        band_counts: Counter[str] = Counter()
        problem_ids: list[str] = []
        family_ids: list[str] = []

        if name in exact_counts:
            expected = exact_counts[name]
            gates.append(
                GateResult(
                    f"exact_count:{name}",
                    len(rows) == expected,
                    True,
                    f"n={len(rows)} expected={expected}",
                    len(rows),
                )
            )

        for row in rows:
            kind = schema_map.get(name) or _infer_schema_kind(row, "")
            verrs = validate_record(kind, row)
            if verrs:
                schema_errors += 1
            try:
                reject_symbolic(str((row.get("answer_spec") or {}).get("type", "")))
                spec = parse_answer_spec(row["answer_spec"]) if row.get("answer_spec") else None
            except (AnswerSpecError, KeyError, TypeError):
                symbolic_errors += 1
                spec = None
                answer_errors += 1
            if require_complete_metadata:
                for key in ("problem_id", "family_id", "partition", "verifier_version"):
                    if not row.get(key):
                        meta_errors += 1
                        break
            pid = str(row.get("problem_id") or "")
            fid = str(row.get("family_id") or "")
            if pid:
                problem_ids.append(pid)
            if fid:
                family_ids.append(fid)
            domain_counts[str(row.get("domain") or "")] += 1
            template_families[_template_family(row)] += 1
            band = _release_band(row)
            if band:
                band_counts[f"{row.get('domain')}:{band}"] += 1

            response = row.get("response") or ""
            if response:
                parsed = parse_response(response)
                if not parsed.format_ok:
                    format_errors += 1
                token_lens.append(len((parsed.think or "").split()))
                arabic_scores.append(
                    float(score_arabic_text(parsed.think or response)["arabic_ratio"])
                )
                if require_per_row_arabic and not score_arabic_text(
                    parsed.think or response
                ).get("pass"):
                    per_row_arabic_errors += 1
                if require_leak_clean:
                    if leak_policy_score(response) != 0.0:
                        leak_errors += 1
                    elif structural_leak_score(response) != 0.0:
                        leak_errors += 1
                if spec is not None:
                    if not verify_answer(response, spec, from_completion=True).ok:
                        verifier_errors += 1
            else:
                prompt = str(row.get("prompt", ""))
                arabic_scores.append(float(score_arabic_text(prompt)["arabic_ratio"]))
                if require_per_row_arabic and float(score_arabic_text(prompt)["arabic_ratio"]) < 0.8:
                    per_row_arabic_errors += 1
            templates[str(row.get("prompt", ""))[:80]] += 1

            status = _decontam_status(row)
            if require_decontam_clean:
                if not status:
                    missing_decontam += 1
                elif status != "clean":
                    non_clean_decontam += 1

        n = max(1, len(rows))
        gates.append(
            GateResult(
                f"schema_validity:{name}",
                schema_errors == 0,
                True,
                f"errors={schema_errors}",
                schema_errors,
            )
        )
        gates.append(
            GateResult(
                f"answer_spec_validity:{name}",
                answer_errors == 0,
                True,
                f"errors={answer_errors}",
                answer_errors,
            )
        )
        gates.append(
            GateResult(
                f"symbolic_rejected:{name}",
                symbolic_errors == 0,
                True,
                f"errors={symbolic_errors}",
                symbolic_errors,
            )
        )
        gates.append(
            GateResult(
                f"format_validity:{name}",
                format_errors == 0,
                True,
                f"errors={format_errors}",
                format_errors,
            )
        )
        gates.append(
            GateResult(
                f"verifier_validity:{name}",
                verifier_errors == 0,
                True,
                f"errors={verifier_errors}",
                verifier_errors,
            )
        )
        gates.append(
            GateResult(
                f"metadata_complete:{name}",
                meta_errors == 0,
                True,
                f"errors={meta_errors}",
                meta_errors,
            )
        )
        gates.append(
            GateResult(
                f"unique_problem_id:{name}",
                len(problem_ids) == len(set(problem_ids)) and len(problem_ids) == len(rows),
                True,
                f"n={len(problem_ids)} unique={len(set(problem_ids))}",
                len(problem_ids) - len(set(problem_ids)),
            )
        )
        gates.append(
            GateResult(
                f"unique_family_id:{name}",
                len(family_ids) == len(set(family_ids)) and len(family_ids) == len(rows),
                True,
                f"n={len(family_ids)} unique={len(set(family_ids))}",
                len(family_ids) - len(set(family_ids)),
            )
        )
        if require_leak_clean and any(r.get("response") for r in rows):
            gates.append(
                GateResult(
                    f"leak_clean:{name}",
                    leak_errors == 0,
                    True,
                    f"errors={leak_errors}",
                    leak_errors,
                )
            )
        if require_per_row_arabic:
            gates.append(
                GateResult(
                    f"per_row_arabic:{name}",
                    per_row_arabic_errors == 0,
                    True,
                    f"errors={per_row_arabic_errors}",
                    per_row_arabic_errors,
                )
            )
        if require_decontam_clean:
            gates.append(
                GateResult(
                    f"decontam_present_clean:{name}",
                    missing_decontam == 0 and non_clean_decontam == 0,
                    True,
                    f"missing={missing_decontam}, non_clean={non_clean_decontam}",
                    missing_decontam + non_clean_decontam,
                )
            )
        if arabic_scores:
            scores = sorted(arabic_scores)
            median = scores[len(scores) // 2]
            p5 = scores[max(0, int(0.05 * (len(scores) - 1)))]
            gates.append(
                GateResult(
                    f"arabic_median:{name}",
                    median >= min_arabic_median,
                    True,
                    f"median={median:.4f}",
                    median,
                )
            )
            gates.append(
                GateResult(
                    f"arabic_p5:{name}",
                    p5 >= min_arabic_p5,
                    True,
                    f"p5={p5:.4f}",
                    p5,
                )
            )
        if token_lens:
            over = sum(1 for t in token_lens if t > token_hard)
            gates.append(
                GateResult(
                    f"token_length_band:{name}",
                    over == 0,
                    True,
                    f"over_hard={over}, soft={token_soft}, hard={token_hard}",
                    over,
                )
            )
        if templates and len(rows) >= 5:
            top_share = templates.most_common(1)[0][1] / n
            gates.append(
                GateResult(
                    f"template_concentration:{name}",
                    top_share <= max_template_share,
                    True,
                    f"top_share={top_share:.4f}",
                    top_share,
                )
            )
        if template_families and len(rows) >= 5:
            # Cap share within each domain.
            by_domain_family: dict[str, Counter[str]] = defaultdict(Counter)
            domain_n: Counter[str] = Counter()
            for row in rows:
                d = str(row.get("domain") or "")
                by_domain_family[d][_template_family(row)] += 1
                domain_n[d] += 1
            worst = 0.0
            for d, fams in by_domain_family.items():
                if domain_n[d] <= 0:
                    continue
                share = fams.most_common(1)[0][1] / domain_n[d]
                worst = max(worst, share)
            gates.append(
                GateResult(
                    f"template_family_cap:{name}",
                    worst <= max_template_family_share + 1e-9,
                    True,
                    f"worst_domain_family_share={worst:.4f}",
                    worst,
                )
            )
        unresolved = sum(
            1
            for r in rows
            if _decontam_status(r) in {"review", "unresolved"}
            or (r.get("metadata") or {}).get("contamination_unresolved")
        )
        gates.append(
            GateResult(
                f"no_unresolved_contamination:{name}",
                unresolved == 0,
                True,
                f"unresolved={unresolved}",
                unresolved,
            )
        )

        if require_domain_quotas and name in {"sft_train", "rlvr_train", "sft", "rlvr"}:
            ok = True
            detail_parts = []
            for domain, need in DOMAIN_QUOTAS.items():
                got = domain_counts.get(domain, 0)
                detail_parts.append(f"{domain}={got}/{need}")
                if got != need:
                    ok = False
            extra = sum(v for k, v in domain_counts.items() if k not in CORE_DOMAINS)
            if extra:
                ok = False
                detail_parts.append(f"extra={extra}")
            gates.append(
                GateResult(
                    f"domain_quotas:{name}",
                    ok,
                    True,
                    ", ".join(detail_parts),
                    extra,
                )
            )

        if require_rlvr_band_matrix and name in {"rlvr_train", "rlvr"} and RLVR_BAND_MATRIX:
            ok = True
            detail_parts = []
            for domain, bands in RLVR_BAND_MATRIX.items():
                for band, need in bands.items():
                    key = f"{domain}:{band}"
                    got = band_counts.get(key, 0)
                    # hard_diagnostic may be labeled hard with release_band
                    if got == 0 and band == "hard_diagnostic":
                        got = band_counts.get(f"{domain}:hard_diagnostic", 0)
                    detail_parts.append(f"{key}={got}/{need}")
                    if got != need:
                        ok = False
            gates.append(
                GateResult(
                    f"rlvr_band_matrix:{name}",
                    ok,
                    True,
                    "; ".join(detail_parts[:12]) + ("..." if len(detail_parts) > 12 else ""),
                    0 if ok else 1,
                )
            )

    gates.extend(audit_family_splits(loaded))
    if manifest is not None:
        registry_matches = manifest.verifier_registry_version == VERIFIER_REGISTRY_VERSION
        gates.append(
            GateResult(
                "verifier_registry_match",
                registry_matches,
                True,
                f"manifest={manifest.verifier_registry_version}, runtime={VERIFIER_REGISTRY_VERSION}",
            )
        )
        if root is not None:
            stale = detect_stale_artifacts(manifest, root)
            gates.append(
                GateResult(
                    "stale_artifacts",
                    len(stale) == 0,
                    True,
                    f"stale={stale}",
                    len(stale),
                )
            )
            if manifest.qa_report_sha256:
                qa_path = root / "qa_report.json"
                if qa_path.exists():
                    ok = sha256_file(qa_path) == manifest.qa_report_sha256
                    gates.append(
                        GateResult(
                            "qa_report_hash",
                            ok,
                            True,
                            "qa_report hash match" if ok else "mismatch",
                        )
                    )
    hard_failures = [g for g in gates if g.hard_failure and (not g.passed)]
    for g in hard_failures:
        errors.append(f"{g.name}: {g.detail}")
    return ShipGateReport(passed=len(hard_failures) == 0, gates=gates, errors=errors)


def write_ship_report(report: ShipGateReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
