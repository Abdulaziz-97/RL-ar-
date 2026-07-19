"""JSON Schema helpers and ship-gate schema validation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

SCHEMA_FILES = {
    "problem": "problem.schema.json",
    "sft_trace": "sft_trace.schema.json",
    "rlvr_prompt": "rlvr_prompt.schema.json",
    "evaluation": "evaluation.schema.json",
}


@lru_cache(maxsize=8)
def load_schema(kind: str) -> dict[str, Any]:
    if kind not in SCHEMA_FILES:
        raise KeyError(f"unknown schema kind: {kind}")
    path = SCHEMA_DIR / SCHEMA_FILES[kind]
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(kind: str, record: dict[str, Any]) -> list[str]:
    """Validate a record. Uses jsonschema if installed; else structural checks."""
    errors: list[str] = []
    schema = load_schema(kind)
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
            errors.append(f"{list(err.path)}: {err.message}")
        return errors
    except ImportError:
        pass

    # Minimal fail-closed fallback without jsonschema dependency.
    for req in schema.get("required", []):
        if req not in record:
            errors.append(f"missing required field: {req}")
    ans = record.get("answer_spec")
    if isinstance(ans, dict):
        t = str(ans.get("type", "")).lower()
        if t == "symbolic" or t.startswith("symbolic"):
            errors.append("symbolic answer_spec rejected")
        allowed = {"integer", "rational", "decimal_exact", "decimal_approx", "logic_json"}
        if t and t not in allowed:
            errors.append(f"unsupported answer type: {t}")
        if "canonical" not in ans:
            errors.append("answer_spec.canonical missing")
    return errors
