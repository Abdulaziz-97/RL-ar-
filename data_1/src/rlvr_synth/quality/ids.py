"""Stable content-derived problem/family identifiers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


def normalize_for_id(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text.strip().casefold())
    return text


def _digest(parts: list[str], *, nbytes: int = 12) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:nbytes]


def content_problem_id(
    *,
    domain: str,
    prompt: str,
    answer_canonical: Any,
    partition: str = "",
    seed: int | None = None,
) -> str:
    canon = (
        answer_canonical
        if isinstance(answer_canonical, str)
        else repr(answer_canonical)
    )
    parts = [
        "v4pid",
        str(domain or ""),
        str(partition or ""),
        normalize_for_id(prompt),
        normalize_for_id(canon),
    ]
    if seed is not None:
        parts.append(f"seed:{int(seed)}")
    return f"prob_{domain}_{_digest(parts)}"


def content_family_id(
    *,
    domain: str,
    prompt: str,
    template_family: str = "",
    seed: int | None = None,
) -> str:
    parts = [
        "v4fid",
        str(domain or ""),
        str(template_family or ""),
        normalize_for_id(prompt),
    ]
    if seed is not None:
        parts.append(f"seed:{int(seed)}")
    return f"fam_{domain}_{_digest(parts)}"
