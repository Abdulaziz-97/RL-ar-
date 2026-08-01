"""Fast decontamination: exact-hash + LSH/MinHash near-dup (linear in batch size)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _record_text(record: dict[str, Any], *, response_chars: int = 800) -> str:
    """Prompt + short response prefix — near-dup signal without hashing full CoT."""
    prompt = str(record.get("prompt") or "").strip()
    response = str(record.get("response") or "").strip()
    if response_chars > 0 and len(response) > response_chars:
        response = response[:response_chars]
    return "\n".join(part for part in (prompt, response) if part)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = normalize_text(text)
    if len(t) < n:
        return {t} if t else set()
    if len(t) > 2500:
        t = t[:2500]
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class MinHashSketch:
    num_perm: int = 32
    seed: int = 0

    def sketch(self, text: str) -> list[int]:
        grams = char_ngrams(text)
        if not grams:
            return [0] * self.num_perm
        # Non-crypto keyed hash — decontam only needs stable within-process sketches.
        out: list[int] = []
        for i in range(self.num_perm):
            best = 0xFFFFFFFF
            salt = (self.seed + 0x9E3779B9 * (i + 1)) & 0xFFFFFFFF
            for g in grams:
                val = (salt ^ (hash(g) & 0xFFFFFFFF)) & 0xFFFFFFFF
                # mix
                val = (val * 0x85EBCA77) & 0xFFFFFFFF
                if val < best:
                    best = val
            out.append(best)
        return out

    def estimate_jaccard(self, a: list[int], b: list[int]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)


class LSHIndex:
    def __init__(self, bands: int = 16, rows: int = 4):
        self.bands = bands
        self.rows = rows
        self.buckets: dict[tuple[int, int], list[str]] = {}
        self.sketches: dict[str, list[int]] = {}

    def add(self, key: str, sketch: list[int]) -> None:
        self.sketches[key] = sketch
        for b in range(self.bands):
            start = b * self.rows
            band = tuple(sketch[start : start + self.rows])
            self.buckets.setdefault((b, hash(band)), []).append(key)

    def query(self, sketch: list[int]) -> set[str]:
        hits: set[str] = set()
        for b in range(self.bands):
            start = b * self.rows
            band = tuple(sketch[start : start + self.rows])
            hits.update(self.buckets.get((b, hash(band)), []))
        return hits


@dataclass
class DecontamResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    exact_hash: str = ""
    max_jaccard: float = 0.0
    max_minhash: float = 0.0
    benchmark_hits: list[str] = field(default_factory=list)


DEFAULT_BENCHMARK_REGISTRY: list[str] = ["gsm8k", "hendrycks math", "arabicmmlu"]


def check_benchmark_registry(
    text: str, registry: Iterable[str] | None = None
) -> list[str]:
    norm = normalize_text(text)
    reg = list(registry) if registry is not None else DEFAULT_BENCHMARK_REGISTRY
    hits = []
    for r in reg:
        token = normalize_text(r)
        if not token:
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", norm):
            hits.append(r)
    return hits


Adjudicator = Callable[[dict[str, Any], DecontamResult], str]


def default_adjudicator(record: dict[str, Any], result: DecontamResult) -> str:
    _ = record
    if result.status in {"reject", "clean"}:
        return result.status
    return "unresolved"


def decontaminate_records(
    records: list[dict[str, Any]],
    *,
    reference_texts: list[str] | None = None,
    jaccard_reject: float = 0.92,
    jaccard_review: float = 0.8,
    minhash_reject: float = 0.9,
    adjudicator: Adjudicator | None = None,
    benchmark_registry: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Near-linear decontam: exact hash + LSH candidates only (no O(n^2) rescans)."""
    adjudicator = adjudicator or default_adjudicator
    mh = MinHashSketch()
    lsh = LSHIndex()
    ref_texts = reference_texts or []
    ref_hashes = {exact_hash(t) for t in ref_texts}
    # Reference n-grams only when few; otherwise rely on MinHash/LSH.
    use_ref_jaccard = 0 < len(ref_texts) <= 2000
    ref_grams = [char_ngrams(t) for t in ref_texts] if use_ref_jaccard else []
    ref_sketches = [mh.sketch(t) for t in ref_texts]
    for i, sk in enumerate(ref_sketches):
        lsh.add(f"ref:{i}", sk)

    seen_hashes: dict[str, str] = {}
    batch_family: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        text = _record_text(rec)
        h = exact_hash(text)
        reasons: list[str] = []
        status = "clean"
        max_jac = 0.0
        max_mh = 0.0
        pid = str(rec.get("problem_id") or "")
        fid = str(rec.get("family_id") or "")

        if h in ref_hashes:
            status = "reject"
            reasons.append("exact_hash_duplicate")
        elif h in seen_hashes and seen_hashes[h] != (pid or fid):
            status = "reject"
            reasons.append("exact_hash_duplicate")
        if h not in seen_hashes:
            seen_hashes[h] = pid or fid or str(idx)

        if use_ref_jaccard:
            grams = char_ngrams(text)
            for rg in ref_grams:
                max_jac = max(max_jac, jaccard(grams, rg))
            if max_jac >= jaccard_reject:
                status = "reject"
                reasons.append(f"jaccard>={jaccard_reject}")
            elif max_jac >= jaccard_review and status == "clean":
                status = "review"
                reasons.append(f"jaccard>={jaccard_review}")

        sk = mh.sketch(text)
        reference_minhash_hit = False
        cross_family_hit = False
        # LSH candidates only — never scan the full batch.
        for other in lsh.query(sk):
            prev = lsh.sketches.get(other)
            if prev is None:
                continue
            similarity = mh.estimate_jaccard(sk, prev)
            max_mh = max(max_mh, similarity)
            if similarity < minhash_reject:
                continue
            if other.startswith("ref:"):
                reference_minhash_hit = True
            elif other.startswith("batch:"):
                other_fid = batch_family.get(other, "")
                if fid and other_fid and other_fid != fid:
                    cross_family_hit = True

        key = f"batch:{idx}"
        lsh.add(key, sk)
        batch_family[key] = fid

        if reference_minhash_hit and "exact_hash_duplicate" not in reasons:
            status = "reject"
            reasons.append("minhash_reference")
        elif cross_family_hit and "exact_hash_duplicate" not in reasons:
            status = "reject"
            reasons.append("minhash_cross_family")

        bench = check_benchmark_registry(text, benchmark_registry)
        if bench:
            status = "review" if status == "clean" else status
            reasons.append("benchmark_registry")

        result = DecontamResult(
            status=status,
            reasons=reasons,
            exact_hash=h,
            max_jaccard=max_jac,
            max_minhash=max_mh,
            benchmark_hits=bench,
        )
        result.status = adjudicator(rec, result)
        results.append(asdict(result))
    return results
