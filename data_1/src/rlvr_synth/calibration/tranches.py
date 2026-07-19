from __future__ import annotations
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

@dataclass
class TrancheMergeResult:
    out_path: str
    n_rows: int
    source_hashes: dict[str, str]
    cache_hit: bool

def merge_tranches(sources: list[Path], out_path: Path, *, cache_dir: Path | None=None, dedupe_on: str='problem_id') -> TrancheMergeResult:
    cache_dir = cache_dir or out_path.parent / '.tranche_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p): _sha256_file(p) for p in sources}
    cache_key = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode('utf-8')).hexdigest()
    cache_meta = cache_dir / f'{cache_key}.json'
    cache_data = cache_dir / f'{cache_key}.jsonl'
    if cache_meta.exists() and cache_data.exists():
        cache_data.replace(out_path) if False else None
        rows = _read_jsonl(cache_data)
        _write_jsonl(out_path, rows)
        return TrancheMergeResult(out_path=str(out_path), n_rows=len(rows), source_hashes=source_hashes, cache_hit=True)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for src in sources:
        for row in _read_jsonl(src):
            key = str(row.get(dedupe_on) or json.dumps(row, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    _write_jsonl(out_path, merged)
    _write_jsonl(cache_data, merged)
    cache_meta.write_text(json.dumps({'source_hashes': source_hashes, 'n': len(merged)}, indent=2), encoding='utf-8')
    return TrancheMergeResult(out_path=str(out_path), n_rows=len(merged), source_hashes=source_hashes, cache_hit=False)

@dataclass
class ExperimentComparison:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    n_items: int
    per_domain: dict[str, float] = field(default_factory=dict)
    stop_recommend: bool = False
    reasons: list[str] = field(default_factory=list)

def paired_bootstrap_ci(deltas: list[float], *, n_boot: int=1000, seed: int=0, alpha: float=0.05) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)

def compare_experiments(results_a: list[dict[str, Any]], results_b: list[dict[str, Any]], *, metric_key: str='correct', id_key: str='problem_id', domain_key: str='domain', seeds: int=3, stop_if_ci_covers_zero: bool=True, stop_if_domain_regression: float=-0.05) -> ExperimentComparison:

    def _agg(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[float]] = {}
        domains: dict[str, str] = {}
        for r in rows:
            pid = str(r[id_key])
            buckets.setdefault(pid, []).append(float(r.get(metric_key, 0.0)))
            domains[pid] = str(r.get(domain_key, ''))
        out = {}
        for pid, vals in buckets.items():
            out[pid] = {'mean': sum(vals[:seeds]) / max(1, len(vals[:seeds])), 'domain': domains[pid]}
        return out
    a = _agg(results_a)
    b = _agg(results_b)
    common = sorted(set(a) & set(b))
    deltas = [b[pid]['mean'] - a[pid]['mean'] for pid in common]
    mean_a = sum((a[pid]['mean'] for pid in common)) / max(1, len(common))
    mean_b = sum((b[pid]['mean'] for pid in common)) / max(1, len(common))
    delta = mean_b - mean_a
    lo, hi = paired_bootstrap_ci(deltas, seed=0)
    per_domain: dict[str, float] = {}
    domain_items: dict[str, list[float]] = {}
    for pid in common:
        domain_items.setdefault(a[pid]['domain'], []).append(b[pid]['mean'] - a[pid]['mean'])
    for dom, vals in domain_items.items():
        per_domain[dom] = sum(vals) / len(vals)
    reasons: list[str] = []
    stop = False
    if stop_if_ci_covers_zero and lo <= 0.0 <= hi and (delta <= 0.0):
        stop = True
        reasons.append('bootstrap_ci_covers_zero_without_gain')
    for dom, d in per_domain.items():
        if d <= stop_if_domain_regression:
            stop = True
            reasons.append(f'domain_regression:{dom}={d:.4f}')
    return ExperimentComparison(metric=metric_key, mean_a=mean_a, mean_b=mean_b, delta=delta, ci_low=lo, ci_high=hi, n_items=len(common), per_domain=per_domain, stop_recommend=stop, reasons=reasons)

def retire_contaminated_eval(eval_path: Path, out_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(eval_path)
    out = []
    for r in rows:
        r = {**r, 'partition': 'contaminated_diagnostic'}
        meta = dict(r.get('metadata') or {})
        meta['retired_reason'] = 'train_eval_or_cross_stage_contamination'
        r['metadata'] = meta
        out.append(r)
    _write_jsonl(out_path, out)
    return {'n': len(out), 'out': str(out_path)}
