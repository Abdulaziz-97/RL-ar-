"""Structured instruction-constraint checkers for IFEval-like training rows.

Uses explicit ``params`` (not prompt parsing) so verification is deterministic
and independent of the pinned AraIFEval benchmark implementation.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+\S", re.MULTILINE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?۔؟])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _bullet_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _BULLET_RE.match(line)]


def check_constraint(category: str, text: str, params: Mapping[str, Any] | None = None) -> bool:
    """Return True iff ``text`` satisfies one structured constraint."""
    params = dict(params or {})
    key = str(category).strip().lower()

    if key in {"word_count_range", "number_words_range"}:
        words = _words(text)
        lo = int(params.get("min", 0))
        hi = int(params.get("max", 10**9))
        return lo <= len(words) <= hi

    if key in {"number_words_at_least", "min_words"}:
        return len(_words(text)) >= int(params["count"])

    if key in {"number_words_at_most", "max_words"}:
        return len(_words(text)) <= int(params["count"])

    if key in {"exact_bullets", "number_bullets"}:
        count = int(params["count"])
        actual = len(_bullet_lines(text))
        if bool(params.get("at_least")):
            return actual >= count
        return actual == count

    if key in {"number_paragraphs", "exact_paragraphs"}:
        count = int(params["count"])
        actual = len(_paragraphs(text))
        if bool(params.get("at_least")):
            return actual >= count
        return actual == count

    if key in {"number_sentences_at_least", "min_sentences"}:
        return len(_sentences(text)) >= int(params["count"])

    if key in {"number_sentences_at_most", "max_sentences"}:
        return len(_sentences(text)) <= int(params["count"])

    if key in {"forbidden_substring", "forbidden_word", "keyword_forbidden"}:
        needle = str(params.get("text") or params.get("word") or "").strip()
        if not needle:
            return False
        return needle not in text

    if key in {"required_substring", "must_include"}:
        needle = str(params.get("text") or "").strip()
        return bool(needle) and needle in text

    if key in {"endswith_line_prefix", "postscript"}:
        prefix = str(params.get("prefix") or "ملاحظة:").strip()
        last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        return last.startswith(prefix)

    if key == "title":
        return bool(re.search(r"<<[^<>\n]+>>", text))

    if key == "quotation":
        t = text.strip()
        return len(t) >= 2 and t[0] in '"“«' and t[-1] in '"”»'

    if key == "response_language":
        lang = str(params.get("language") or "ar").lower()
        if lang in {"en", "english", "الإنجليزية", "الانجليزية"}:
            return bool(_LATIN_RE.search(text)) and not _ARABIC_RE.search(text)
        return bool(_ARABIC_RE.search(text)) and not _LATIN_RE.search(text)

    if key in {"placeholder_count", "number_placeholder"}:
        count = int(params["count"])
        return len(re.findall(r"\[[^\[\]\n]+\]", text)) >= count

    raise KeyError(f"Unsupported constraint category: {category}")


def check_all_constraints(
    text: str,
    constraints: list[Mapping[str, Any]],
    *,
    pass_all: bool = True,
) -> tuple[bool, list[bool]]:
    results: list[bool] = []
    for item in constraints:
        category = str(item.get("category") or "")
        params = item.get("params") or {}
        results.append(check_constraint(category, text, params))
    if not results:
        return False, []
    ok = all(results) if pass_all else any(results)
    return ok, results


def strip_think_answer_wrappers(text: str) -> str:
    """Prefer <answer> body when present; otherwise drop <think> blocks."""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip() or text.strip()
