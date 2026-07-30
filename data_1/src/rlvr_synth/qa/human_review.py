from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

@dataclass
class ReviewItem:
    item_id: str
    stratum: str
    prompt: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict)

def stratified_sample(records: list[dict[str, Any]], *, n: int, strata_keys: tuple[str, ...]=('domain', 'partition'), seed: int=0, oversample_hard: bool=True) -> list[ReviewItem]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = '|'.join((str(r.get(k, '')) for k in strata_keys))
        buckets[key].append(r)
    strata = list(buckets.keys())
    if not strata:
        return []
    per = max(1, n // len(strata))
    chosen: list[dict[str, Any]] = []
    for key in strata:
        pool = list(buckets[key])
        if oversample_hard:
            pool.sort(key=lambda r: (0 if (r.get('empirical_difficulty') or {}).get('band') == 'hard' or r.get('difficulty_tag') == 'hard' else 1, rng.random()))
        else:
            rng.shuffle(pool)
        chosen.extend(pool[:per])
    rng.shuffle(chosen)
    chosen = chosen[:n]
    items = []
    for i, r in enumerate(chosen):
        items.append(ReviewItem(item_id=str(r.get('problem_id') or f'item_{i}'), stratum='|'.join((str(r.get(k, '')) for k in strata_keys)), prompt=str(r.get('prompt', '')), response=str(r.get('response', '')), meta={'domain': r.get('domain'), 'partition': r.get('partition'), 'family_id': r.get('family_id')}))
    return items

def export_review_pack(items: list[ReviewItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        rows.append({**asdict(it), 'annotations': [], 'adjudication': None})
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

def import_review_pack(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))

def wilson_interval(successes: int, n: int, z: float=1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return (max(0.0, low), min(1.0, high))

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    if len(y1) != len(y2) or not y1:
        return 0.0
    n = len(y1)
    labels = sorted(set(y1) | set(y2))
    po = sum((1 for a, b in zip(y1, y2) if a == b)) / n
    pe = 0.0
    for lab in labels:
        pe += sum((1 for a in y1 if a == lab)) / n * (sum((1 for b in y2 if b == lab)) / n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

def weighted_kappa(y1: list[int], y2: list[int], weights: str='linear') -> float:
    if len(y1) != len(y2) or not y1:
        return 0.0
    labels = sorted(set(y1) | set(y2))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    o = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        o[idx[a]][idx[b]] += 1
    n = float(len(y1))
    row = [sum(r) / n for r in o]
    col = [sum((o[i][j] for i in range(k))) / n for j in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j)
        if weights == 'quadratic':
            return 1.0 - (d / (k - 1)) ** 2 if k > 1 else 1.0
        return 1.0 - d / (k - 1) if k > 1 else 1.0
    po = sum((w(i, j) * o[i][j] / n for i in range(k) for j in range(k)))
    pe = sum((w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

def summarize_reviews(pack: list[dict[str, Any]], *, min_accept_rate: float=0.95, min_kappa: float=0.6, min_wilson_low: float=0.9) -> dict[str, Any]:
    accepted = 0
    total = 0
    a1: list[int] = []
    a2: list[int] = []
    for item in pack:
        anns = item.get('annotations') or []
        if not anns:
            continue
        labels = [1 if a.get('label') in (1, 'accept', True) else 0 for a in anns]
        final = item.get('adjudication')
        if final is None:
            final = 1 if sum(labels) >= (len(labels) + 1) // 2 else 0
        else:
            final = 1 if final in (1, 'accept', True) else 0
        total += 1
        accepted += int(final)
        if len(labels) >= 2:
            a1.append(labels[0])
            a2.append(labels[1])
    low, high = wilson_interval(accepted, total)
    kappa = cohen_kappa(a1, a2) if a1 else None
    wkappa = weighted_kappa(a1, a2) if a1 else None
    accept_rate = accepted / total if total else 0.0
    hard_pass = total > 0 and accept_rate >= min_accept_rate and (low >= min_wilson_low) and (kappa is None or kappa >= min_kappa)
    return {'n': total, 'accepted': accepted, 'accept_rate': accept_rate, 'wilson': {'low': low, 'high': high}, 'cohen_kappa': kappa, 'weighted_kappa': wkappa, 'pass': hard_pass, 'thresholds': {'min_accept_rate': min_accept_rate, 'min_kappa': min_kappa, 'min_wilson_low': min_wilson_low}}
