"""Formal Arabic (FusHa / MSA) style lock — no regional dialects."""
from __future__ import annotations

import re

FORMAL_ARABIC_STYLE = """
[Language Style — Formal Arabic / الفصحى فقط (mandatory)]
Write ONLY in clear Modern Standard Arabic (FusHa): correct grammar, formal vocabulary, complete sentences.

STRICTLY FORBIDDEN — do not use ANY regional dialect:
- Saudi / Gulf colloquial: وش، أبغى، دحين، يبي، كمان، زين، يلا، ترى، مو (as dialect), اللي (as dialect filler), أيش، هالحين، بروح، عشان (prefer لأن / من أجل)
- Egyptian: عايز، مش، إزاي، كده، دلوقتي، هعمل، بتاع، أوي، خلاص (colloquial sense)
- Levantine (Syrian/Lebanese/Palestinian/Jordanian): شو، هلق، بدّي، منيح، ليش، عمّال، هيك، يلا
- Iraqi: شلون، أكو، ماكو، وين
- Maghrebi / North African: بزاف، واش، كاين، باش، دابا
- Yemeni / Sudanese colloquial markers and similar

Use MSA alternatives: ماذا / ما / كم / هل؛ الذي / التي؛ الآن؛ أيضًا / كذلك؛ لأن؛ ليس؛ كيف؛ لماذا؛ أين.
Contexts must be global/non-local (not tied to a specific Arab country's slang or geography).
One optional light particle is NEVER required — prefer pure FusHa.
""".strip()

# Back-compat alias
FORMAL_SAUDI_STYLE = FORMAL_ARABIC_STYLE

# All dialect markers to reject (any hit counts toward colloquial score)
DIALECT_MARKERS = [
    # Saudi / Gulf
    "دحين",
    "أبغى",
    "كمان",
    "زين",
    "أيش",
    "وش",
    "يلا",
    "ترى",
    "يبي",
    "هالحين",
    "وش رايك",
    "أبا",
    "أبي",
    # Egyptian — avoid short substrings that appear inside MSA words (e.g. زاوية→اوي)
    "عايز",
    "عاوز",
    "دلوقتي",
    "إزاي",
    "ازاي",
    "كده",
    "بتاع",
    "هعمل",
    "مش عارف",
    # Levantine
    "هلق",
    "هلأ",
    "بدّي",
    "بدي",
    "منيح",
    "ليش",
    "هيك",
    "عمّال",
    "شو",
    # Iraqi
    "شلون",
    "أكو",
    "ماكو",
    # Maghrebi
    "بزاف",
    "واش",
    "كاين",
    "دابا",
]

# Soft dialect-ish (penalize if frequent; MSA often uses اللي too — only flag with other dialect)
SOFT_DIALECT = ["مو ", "اللي ", "عشان "]


def arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = sum(
        1
        for ch in text
        if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff"
    )
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = arabic + latin
    return arabic / total if total else 1.0


def count_occurrences(text: str, terms: list[str]) -> int:
    return sum(text.count(t) for t in terms)


def dialect_hits(text: str) -> list[str]:
    """Return dialect markers found as whole tokens (avoid substring false positives)."""
    if not text:
        return []
    found = []
    for m in DIALECT_MARKERS:
        term = m.strip()
        if not term:
            continue
        # Word-boundary style: not preceded/followed by Arabic/Latin letters
        pat = rf"(?<![\u0600-\u06ffA-Za-z]){re.escape(term)}(?![\u0600-\u06ffA-Za-z])"
        if re.search(pat, text):
            found.append(term)
    return found


def formal_saudi_score(prompt: str, domain: str = "") -> tuple[float, list[str]]:
    """Score formal MSA quality (name kept for back-compat). Higher is better."""
    issues = []
    if not prompt:
        return 0.0, ["empty_prompt"]

    hits = dialect_hits(prompt)
    soft = count_occurrences(prompt, SOFT_DIALECT)
    ar = arabic_char_ratio(prompt)

    score = 1.0
    ar_floor = 0.55 if domain == "math_comp" else 0.75
    if ar < ar_floor:
        score -= 0.35
        issues.append(f"arabic_ratio_low:{ar:.2f}")

    if hits:
        score -= min(0.7, 0.25 * len(hits))
        issues.append(f"dialect_markers:{','.join(hits[:8])}")

    if soft >= 3:
        score -= 0.25
        issues.append(f"soft_dialect_spam:{soft}")
    elif soft >= 2:
        score -= 0.1
        issues.append(f"soft_dialect:{soft}")

    formal_cues = sum(
        prompt.count(x)
        for x in ["الذي", "التي", "ما هو", "كم ", "إذا كان", "أوجد", "احسب", "فإن", "هل "]
    )
    if formal_cues == 0 and domain not in ("math_comp", "mmlu") and len(prompt.split()) > 25:
        score -= 0.1
        issues.append("weak_formal_cues")

    return max(0.0, min(1.0, score)), issues


def passes_formal_saudi(prompt: str, domain: str = "", min_score: float = 0.6) -> bool:
    score, _ = formal_saudi_score(prompt, domain)
    return score >= min_score


def has_literal_backslash_n(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"(?<![A-Za-z\\])\\[nt](?![A-Za-z])", text))


def tokenize_arabic(text: str) -> list[str]:
    return re.findall(r"[\u0600-\u06ff\u0750-\u077f0-9A-Za-z]+", text or "")


def unique_ratio_think(response: str) -> float:
    m = re.search(r"<think>(.*?)</think>", response or "", re.DOTALL | re.IGNORECASE)
    if not m:
        return 1.0
    tokens = m.group(1).split()
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def jaccard(a: str, b: str) -> float:
    wa, wb = set(tokenize_arabic(a)), set(tokenize_arabic(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
