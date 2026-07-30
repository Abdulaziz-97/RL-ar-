from __future__ import annotations
import re
from typing import Any
_ARABIC_RE = re.compile('[\\u0600-\\u06FF]')
_LATIN_RE = re.compile('[A-Za-z]')
_DIALECT_MARKERS = ('بيقول', 'عايز', 'هعمل', 'مش عارف', 'ليش', 'هيك')
_FOREIGN_PROSE = ('as an ai', 'according to', 'in conclusion')

def _strip_equations_and_json(text: str) -> str:
    text = re.sub('\\$[^$]+\\$', ' ', text)
    text = re.sub('\\{[^{}]*\\}', ' ', text)
    text = re.sub('\\[[^\\[\\]]*\\]', ' ', text)
    text = re.sub('[\\d\\W_]+', ' ', text, flags=re.UNICODE)
    return text

def arabic_char_ratio(text: str) -> float:
    prose = _strip_equations_and_json(text)
    letters = [c for c in prose if _ARABIC_RE.search(c) or _LATIN_RE.search(c)]
    if not letters:
        return 1.0
    ar = sum((1 for c in letters if _ARABIC_RE.search(c)))
    return ar / len(letters)

def foreign_prose_hits(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _FOREIGN_PROSE if p in low]

def dialect_hits(text: str) -> list[str]:
    return [m for m in _DIALECT_MARKERS if m in text]

def unicode_anomaly_score(text: str) -> float:
    if not text:
        return 0.0
    bad = sum((1 for c in text if ord(c) < 32 and c not in '\n\t\r'))
    return bad / max(1, len(text))

def repetition_score(text: str) -> float:
    toks = text.split()
    if len(toks) < 4:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)

def terminology_consistency(text: str) -> float:
    if re.search('\\b(answer|solution|proof)\\b', text.lower()) and _ARABIC_RE.search(text):
        return 0.5
    return 1.0

def score_arabic_text(text: str, *, min_ratio: float=0.85, max_repetition: float=0.55, calibration: dict[str, Any] | None=None) -> dict[str, Any]:
    cal = calibration or {}
    min_ratio = float(cal.get('min_ratio', min_ratio))
    max_repetition = float(cal.get('max_repetition', max_repetition))
    prose = re.sub('</?(?:think|answer)>', ' ', text, flags=re.I)
    ratio = arabic_char_ratio(prose)
    foreign = foreign_prose_hits(prose)
    dialect = dialect_hits(prose)
    uni = unicode_anomaly_score(prose)
    rep = repetition_score(prose)
    term = terminology_consistency(prose)
    passed = ratio >= min_ratio and (not foreign) and (not dialect) and (uni < 0.01) and (rep <= max_repetition) and (term >= 0.5)
    return {'arabic_ratio': ratio, 'foreign_prose': foreign, 'dialect': dialect, 'unicode_anomaly': uni, 'repetition': rep, 'terminology': term, 'pass': passed, 'calibration': {'min_ratio': min_ratio, 'max_repetition': max_repetition}}

def score_arabic_record(record: dict[str, Any], calibration: dict[str, Any] | None=None) -> dict[str, Any]:
    response = str(record.get('response') or '')
    if response:
        from rlvr_contracts.response import extract_think
        think = extract_think(response)
        text = think if think else response
    else:
        text = str(record.get('prompt') or '')
    return score_arabic_text(text, calibration=calibration)

def release_arabic_gates(scores: list[dict[str, Any]], *, min_median: float=0.9, min_p5: float=0.8) -> dict[str, Any]:
    if not scores:
        return {'pass': False, 'reason': 'empty'}
    ratios = sorted((float(s.get('arabic_ratio', 0.0)) for s in scores))
    median = ratios[len(ratios) // 2]
    p5 = ratios[max(0, int(0.05 * (len(ratios) - 1)))]
    return {'pass': median >= min_median and p5 >= min_p5, 'median': median, 'p5': p5, 'min_median': min_median, 'min_p5': min_p5}
