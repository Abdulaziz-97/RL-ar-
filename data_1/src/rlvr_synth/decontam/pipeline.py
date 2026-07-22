from __future__ import annotations
import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text or '')
    text = text.replace('ى', 'ي').replace('ة', 'ه')
    text = re.sub('\\s+', ' ', text.strip().lower())
    return text

def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()

def _record_text(record: dict[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            str(record.get("prompt") or "").strip(),
            str(record.get("response") or "").strip(),
        )
        if value
    )

def char_ngrams(text: str, n: int=5) -> set[str]:
    t = normalize_text(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and (not b):
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

@dataclass
class MinHashSketch:
    num_perm: int = 64
    seed: int = 0

    def sketch(self, text: str) -> list[int]:
        grams = char_ngrams(text)
        if not grams:
            return [0] * self.num_perm
        out = []
        for i in range(self.num_perm):
            best = None
            for g in grams:
                h = hashlib.sha1(f'{self.seed}:{i}:{g}'.encode('utf-8')).hexdigest()
                val = int(h[:8], 16)
                if best is None or val < best:
                    best = val
            out.append(int(best or 0))
        return out

    def estimate_jaccard(self, a: list[int], b: list[int]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum((1 for x, y in zip(a, b) if x == y)) / len(a)

class LSHIndex:

    def __init__(self, bands: int=16, rows: int=4):
        self.bands = bands
        self.rows = rows
        self.buckets: dict[tuple[int, int], list[str]] = {}
        self.sketches: dict[str, list[int]] = {}

    def add(self, key: str, sketch: list[int]) -> None:
        self.sketches[key] = sketch
        for b in range(self.bands):
            start = b * self.rows
            band = tuple(sketch[start:start + self.rows])
            self.buckets.setdefault((b, hash(band)), []).append(key)

    def query(self, sketch: list[int]) -> set[str]:
        hits: set[str] = set()
        for b in range(self.bands):
            start = b * self.rows
            band = tuple(sketch[start:start + self.rows])
            hits.update(self.buckets.get((b, hash(band)), []))
        return hits

@dataclass
class DecontamResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    exact_hash: str = ''
    max_jaccard: float = 0.0
    max_minhash: float = 0.0
    benchmark_hits: list[str] = field(default_factory=list)
DEFAULT_BENCHMARK_REGISTRY: list[str] = ['gsm8k', 'hendrycks math', 'arabicmmlu']

def check_benchmark_registry(text: str, registry: Iterable[str] | None=None) -> list[str]:
    norm = normalize_text(text)
    reg = list(registry) if registry is not None else DEFAULT_BENCHMARK_REGISTRY
    hits = []
    for r in reg:
        token = normalize_text(r)
        if not token:
            continue
        if re.search(f'(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])', norm):
            hits.append(r)
    return hits
Adjudicator = Callable[[dict[str, Any], DecontamResult], str]

def default_adjudicator(record: dict[str, Any], result: DecontamResult) -> str:
    _ = record
    if result.status in {'reject', 'clean'}:
        return result.status
    return 'unresolved'

def decontaminate_records(records: list[dict[str, Any]], *, reference_texts: list[str] | None=None, jaccard_reject: float=0.92, jaccard_review: float=0.8, minhash_reject: float=0.9, adjudicator: Adjudicator | None=None, benchmark_registry: list[str] | None=None) -> list[dict[str, Any]]:
    adjudicator = adjudicator or default_adjudicator
    mh = MinHashSketch()
    lsh = LSHIndex()
    ref_texts = reference_texts or []
    ref_hashes = {exact_hash(t) for t in ref_texts}
    ref_sketches = [mh.sketch(t) for t in ref_texts]
    for i, sk in enumerate(ref_sketches):
        lsh.add(f'ref:{i}', sk)
    seen_hashes: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        text = _record_text(rec)
        h = exact_hash(text)
        reasons: list[str] = []
        status = 'clean'
        max_jac = 0.0
        max_mh = 0.0
        pid = str(rec.get('problem_id') or '')
        fid = str(rec.get('family_id') or '')
        if h in ref_hashes:
            status = 'reject'
            reasons.append('exact_hash_duplicate')
        elif h in seen_hashes and seen_hashes[h] != (pid or fid):
            status = 'reject'
            reasons.append('exact_hash_duplicate')
        if h not in seen_hashes:
            seen_hashes[h] = pid or fid or str(idx)
        grams = char_ngrams(text)
        for ref in ref_texts:
            max_jac = max(max_jac, jaccard(grams, char_ngrams(ref)))
        if max_jac >= jaccard_reject:
            status = 'reject'
            reasons.append(f'jaccard>={jaccard_reject}')
        elif max_jac >= jaccard_review and status == 'clean':
            status = 'review'
            reasons.append(f'jaccard>={jaccard_review}')
        sk = mh.sketch(text)
        reference_minhash_hit = False
        for other in lsh.query(sk):
            if other.startswith('ref:'):
                ri = int(other.split(':')[1])
                similarity = mh.estimate_jaccard(sk, ref_sketches[ri])
                max_mh = max(max_mh, similarity)
                if similarity >= minhash_reject:
                    reference_minhash_hit = True
        for key, prev in list(lsh.sketches.items()):
            if key.startswith('batch:'):
                max_mh = max(max_mh, mh.estimate_jaccard(sk, prev))
        lsh.add(f'batch:{idx}', sk)
        if reference_minhash_hit and 'exact_hash_duplicate' not in reasons:
            status = 'reject'
            reasons.append('minhash_reference')
        elif max_mh >= minhash_reject and 'exact_hash_duplicate' not in reasons:
            if rec.get('family_id') and any((True for j, other in enumerate(records[:idx]) if other.get('family_id') != rec.get('family_id') and mh.estimate_jaccard(sk, mh.sketch(_record_text(other))) >= minhash_reject)):
                status = 'reject' if status != 'reject' else status
                reasons.append('minhash_cross_family')
        bench = check_benchmark_registry(text, benchmark_registry)
        if bench:
            status = 'review' if status == 'clean' else status
            reasons.append('benchmark_registry')
        result = DecontamResult(status=status, reasons=reasons, exact_hash=h, max_jaccard=max_jac, max_minhash=max_mh, benchmark_hits=bench)
        final_status = adjudicator(rec, result)
        result.status = final_status
        results.append(asdict(result))
    return results
