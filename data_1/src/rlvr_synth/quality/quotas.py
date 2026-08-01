"""Exact domain and RLVR band quotas for V4 releases."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

CORE_DOMAINS = ("gsm8k", "math", "math_comp", "logic")

DOMAIN_QUOTAS = {
    "gsm8k": 1200,
    "math": 1000,
    "math_comp": 1000,
    "logic": 800,
}
SFT_DOMAIN_QUOTAS = dict(DOMAIN_QUOTAS)

# Exact 4,000-row domain × band matrix (hard-diagnostic counted under hard_diagnostic).
RLVR_BAND_MATRIX: dict[str, dict[str, int]] = {
    "gsm8k": {"easy": 360, "medium": 600, "hard": 216, "hard_diagnostic": 24},
    "math": {"easy": 300, "medium": 500, "hard": 180, "hard_diagnostic": 20},
    "math_comp": {"easy": 300, "medium": 500, "hard": 180, "hard_diagnostic": 20},
    "logic": {"easy": 240, "medium": 400, "hard": 144, "hard_diagnostic": 16},
}

USABLE_BANDS = ("easy", "medium", "hard", "hard_diagnostic")
SIDE_BANDS = ("mastered", "deferred")


def select_domain_quota(
    rows: Iterable[dict[str, Any]],
    *,
    quotas: dict[str, int] | None = None,
    domain_key: str = "domain",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically fill exact domain quotas; report shortages."""
    target = quotas or DOMAIN_QUOTAS
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = str(row.get(domain_key) or "")
        if domain in target:
            buckets[domain].append(row)
    for domain in buckets:
        buckets[domain].sort(
            key=lambda r: (
                str(r.get("problem_id") or ""),
                str(r.get("family_id") or ""),
                str(r.get("prompt") or ""),
            )
        )
    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for domain, need in target.items():
        pool = buckets.get(domain, [])
        if len(pool) < need:
            shortages[domain] = need - len(pool)
            selected.extend(pool)
        else:
            selected.extend(pool[:need])
    selected.sort(
        key=lambda r: (
            str(r.get("domain") or ""),
            str(r.get("problem_id") or ""),
        )
    )
    return selected, shortages


def select_rlvr_band_matrix(
    rows: Iterable[dict[str, Any]],
    *,
    matrix: dict[str, dict[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Fill the exact domain×band matrix.

    Returns (selected, mastered, deferred, shortages_by_key).
    hard_diagnostic must come from empirical hard with passes == 1.
    """
    target = matrix or RLVR_BAND_MATRIX
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    mastered: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for row in rows:
        domain = str(row.get("domain") or "")
        emp = row.get("empirical_difficulty") or {}
        meta = row.get("metadata") or {}
        band = str(
            emp.get("band")
            or meta.get("difficulty_band")
            or row.get("difficulty_tag")
            or ""
        )
        passes = emp.get("passes")
        if passes is None:
            passes = (meta.get("pass_at_8") or {}).get("passes")
        if band == "mastered":
            mastered.append(row)
            continue
        if band == "deferred":
            deferred.append(row)
            continue
        if band == "hard" and passes == 1:
            by_key[(domain, "hard_diagnostic")].append(row)
        if band in {"easy", "medium", "hard"}:
            by_key[(domain, band)].append(row)

    for key in by_key:
        by_key[key].sort(
            key=lambda r: (
                str(r.get("problem_id") or ""),
                str(r.get("family_id") or ""),
            )
        )

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    shortages: dict[str, int] = {}

    def _take(domain: str, band: str, need: int) -> None:
        pool = [
            r
            for r in by_key.get((domain, band), [])
            if str(r.get("problem_id") or "") not in used_ids
        ]
        if len(pool) < need:
            shortages[f"{domain}:{band}"] = need - len(pool)
            chosen = pool
        else:
            chosen = pool[:need]
        for row in chosen:
            pid = str(row.get("problem_id") or "")
            used_ids.add(pid)
            out = dict(row)
            out["difficulty_tag"] = band if band != "hard_diagnostic" else "hard"
            emp = dict(out.get("empirical_difficulty") or {})
            emp["release_band"] = band
            out["empirical_difficulty"] = emp
            meta = dict(out.get("metadata") or {})
            meta["difficulty_band"] = band
            meta["difficulty_source"] = meta.get("difficulty_source") or "sft_pass_at_8"
            out["metadata"] = meta
            selected.append(out)

    for domain, bands in target.items():
        # Reserve hard-diagnostic (1/8) before filling the general hard quota.
        for band in ("easy", "medium", "hard_diagnostic", "hard"):
            _take(domain, band, int(bands.get(band, 0)))

    selected.sort(
        key=lambda r: (
            str(r.get("domain") or ""),
            str((r.get("empirical_difficulty") or {}).get("release_band") or ""),
            str(r.get("problem_id") or ""),
        )
    )
    return selected, mastered, deferred, shortages
