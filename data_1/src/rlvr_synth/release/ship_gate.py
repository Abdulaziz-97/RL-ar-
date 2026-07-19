from __future__ import annotations
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from rlvr_contracts.answer_spec import AnswerSpecError, parse_answer_spec, reject_symbolic
from rlvr_contracts.response import parse_response
from rlvr_contracts.schema_validate import validate_record
from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION, verify_answer
from rlvr_synth.release.manifest import ReleaseManifest, detect_stale_artifacts, sha256_file
_ARABIC_RE = re.compile('[\\u0600-\\u06FF]')

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
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _arabic_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha() or _ARABIC_RE.search(c)]
    if not chars:
        return 1.0
    ar = sum((1 for c in chars if _ARABIC_RE.search(c)))
    return ar / len(chars)

def _infer_schema_kind(row: dict[str, Any], default: str) -> str:
    partition = str(row.get('partition', ''))
    if 'access_control' in row or partition in {'private_eval', 'contaminated_diagnostic'}:
        if 'response' in row and row.get('response'):
            return 'sft_trace'
        if partition == 'contaminated_diagnostic' and (not row.get('response')):
            return 'rlvr_prompt'
        return 'evaluation'
    if row.get('response'):
        return 'sft_trace'
    if default:
        return default
    return 'rlvr_prompt'

def audit_family_splits(corpora: dict[str, list[dict[str, Any]]]) -> list[GateResult]:
    results: list[GateResult] = []
    family_sets = {name: {str(r.get('family_id', '')) for r in rows if r.get('family_id')} for name, rows in corpora.items()}
    prompt_sets = {name: {str(r.get('prompt', '')).strip() for r in rows if r.get('prompt')} for name, rows in corpora.items()}
    pairs = [('sft_train', 'rlvr_train'), ('sft_train', 'rlvr_eval'), ('sft_train', 'private_eval'), ('rlvr_train', 'rlvr_eval'), ('rlvr_train', 'private_eval'), ('sft_eval', 'rlvr_train'), ('sft_eval', 'private_eval')]
    for a, b in pairs:
        if a not in family_sets or b not in family_sets:
            continue
        fam_overlap = family_sets[a] & family_sets[b]
        fam_overlap.discard('')
        results.append(GateResult(name=f'family_overlap:{a}|{b}', passed=len(fam_overlap) == 0, hard_failure=True, detail=f'overlap={len(fam_overlap)}', metric=len(fam_overlap)))
        prompt_overlap = prompt_sets[a] & prompt_sets[b]
        prompt_overlap.discard('')
        results.append(GateResult(name=f'prompt_overlap:{a}|{b}', passed=len(prompt_overlap) == 0, hard_failure=True, detail=f'overlap={len(prompt_overlap)}', metric=len(prompt_overlap)))
    return results

def run_ship_gate(*, corpora: dict[str, Path], manifest: ReleaseManifest | None=None, root: Path | None=None, schema_kind_by_corpus: dict[str, str] | None=None, min_arabic_median: float=0.9, min_arabic_p5: float=0.8, token_soft: int=256, token_hard: int=640, max_template_share: float=0.35, require_complete_metadata: bool=True) -> ShipGateReport:
    gates: list[GateResult] = []
    errors: list[str] = []
    loaded: dict[str, list[dict[str, Any]]] = {}
    schema_map = schema_kind_by_corpus or {}
    for name, path in corpora.items():
        rows = _read_jsonl(path)
        loaded[name] = rows
        schema_errors = 0
        answer_errors = 0
        format_errors = 0
        verifier_errors = 0
        symbolic_errors = 0
        meta_errors = 0
        arabic_scores: list[float] = []
        token_lens: list[int] = []
        templates: Counter[str] = Counter()
        for row in rows:
            kind = schema_map.get(name) or _infer_schema_kind(row, 'problem')
            verrs = validate_record(kind, row)
            if verrs:
                schema_errors += 1
            try:
                reject_symbolic(str((row.get('answer_spec') or {}).get('type', '')))
                spec = parse_answer_spec(row['answer_spec']) if row.get('answer_spec') else None
            except (AnswerSpecError, KeyError, TypeError):
                symbolic_errors += 1
                spec = None
                answer_errors += 1
            if require_complete_metadata:
                for key in ('problem_id', 'family_id', 'partition', 'verifier_version'):
                    if not row.get(key):
                        meta_errors += 1
                        break
            response = row.get('response') or ''
            if response:
                parsed = parse_response(response)
                if not parsed.format_ok:
                    format_errors += 1
                token_lens.append(len((parsed.think or '').split()))
                arabic_scores.append(_arabic_ratio(parsed.think or response))
                if spec is not None:
                    if not verify_answer(response, spec, from_completion=True).ok:
                        verifier_errors += 1
            else:
                arabic_scores.append(_arabic_ratio(str(row.get('prompt', ''))))
            templates[str(row.get('prompt', ''))[:80]] += 1
        n = max(1, len(rows))
        gates.append(GateResult('schema_validity', schema_errors == 0, True, f'errors={schema_errors}', schema_errors))
        gates.append(GateResult('answer_spec_validity', answer_errors == 0, True, f'errors={answer_errors}', answer_errors))
        gates.append(GateResult('symbolic_rejected', symbolic_errors == 0, True, f'errors={symbolic_errors}', symbolic_errors))
        gates.append(GateResult('format_validity', format_errors == 0, True, f'errors={format_errors}', format_errors))
        gates.append(GateResult('verifier_validity', verifier_errors == 0, True, f'errors={verifier_errors}', verifier_errors))
        gates.append(GateResult('metadata_complete', meta_errors == 0, True, f'errors={meta_errors}', meta_errors))
        if arabic_scores:
            scores = sorted(arabic_scores)
            median = scores[len(scores) // 2]
            p5 = scores[max(0, int(0.05 * (len(scores) - 1)))]
            gates.append(GateResult(f'arabic_median:{name}', median >= min_arabic_median, True, f'median={median:.4f}', median))
            gates.append(GateResult(f'arabic_p5:{name}', p5 >= min_arabic_p5, True, f'p5={p5:.4f}', p5))
        if token_lens:
            over = sum((1 for t in token_lens if t > token_hard))
            gates.append(GateResult(f'token_length_band:{name}', over == 0, True, f'over_hard={over}, soft={token_soft}, hard={token_hard}', over))
        if templates and len(rows) >= 5:
            top_share = templates.most_common(1)[0][1] / n
            gates.append(GateResult(f'template_concentration:{name}', top_share <= max_template_share, True, f'top_share={top_share:.4f}', top_share))
        unresolved = sum((1 for r in rows if (r.get('decontam') or {}).get('status') == 'unresolved' or (r.get('metadata') or {}).get('contamination_unresolved')))
        gates.append(GateResult(f'no_unresolved_contamination:{name}', unresolved == 0, True, f'unresolved={unresolved}', unresolved))
    gates.extend(audit_family_splits(loaded))
    if manifest is not None:
        gates.append(GateResult('verifier_registry_match', manifest.verifier_registry_version == VERIFIER_REGISTRY_VERSION or bool(manifest.verifier_registry_version), True, f'manifest={manifest.verifier_registry_version}, runtime={VERIFIER_REGISTRY_VERSION}'))
        if root is not None:
            stale = detect_stale_artifacts(manifest, root)
            gates.append(GateResult('stale_artifacts', len(stale) == 0, True, f'stale={stale}', len(stale)))
            if manifest.qa_report_sha256:
                qa_path = root / 'qa_report.json'
                if qa_path.exists():
                    ok = sha256_file(qa_path) == manifest.qa_report_sha256
                    gates.append(GateResult('qa_report_hash', ok, True, 'qa_report hash match' if ok else 'mismatch'))
    hard_failures = [g for g in gates if g.hard_failure and (not g.passed)]
    for g in hard_failures:
        errors.append(f'{g.name}: {g.detail}')
    return ShipGateReport(passed=len(hard_failures) == 0, gates=gates, errors=errors)

def write_ship_report(report: ShipGateReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
