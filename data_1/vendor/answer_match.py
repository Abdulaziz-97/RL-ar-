"""Lightweight answer equality helpers used by the Arabic teacher metric.

- `answers_match_numeric` / `answers_match_logic` → bool
- `answers_match_math_comp` → \"match\" / \"mismatch\" (string status)
"""

from __future__ import annotations

import json
import re
from typing import Any

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(x: Any):
    if isinstance(x, (int, float)) and (not isinstance(x, bool)):
        return float(x)
    if isinstance(x, str):
        m = _NUM.search(x.replace(",", ""))
        if m:
            return float(m.group(0))
    return None


def answers_match_numeric(model_ans, gt) -> bool:
    a = _to_float(model_ans)
    b = _to_float(gt)
    if a is None or b is None:
        return str(model_ans).strip() == str(gt).strip()
    return abs(a - b) < 1e-06


def answers_match_logic(model_ans, gt) -> bool:
    if isinstance(model_ans, str):
        try:
            model_ans = json.loads(model_ans)
        except json.JSONDecodeError:
            pass
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            pass
    return model_ans == gt


def answers_match_math_comp(model_ans, gt) -> str:
    if answers_match_numeric(model_ans, gt):
        return "match"
    sa, sg = str(model_ans).strip(), str(gt).strip()
    return "match" if sa == sg or sa.replace(" ", "") == sg.replace(" ", "") else "mismatch"
