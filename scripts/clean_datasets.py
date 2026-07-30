#!/usr/bin/env python
"""Clean Arabic RLVR / coldstart JSONL for trainable signal.

Fixes the concrete data bugs that zeroed format/correctness in GRPO:
  1. Strip trailing surface-noise parentheses from every prompt
  2. Persist <answer>…</answer> on coldstart (replace #### endings)
  3. Fill empty RLVR top-level `answer` from metadata.ground_truth_answer

Usage (from repo root):
  python scripts/clean_datasets.py
  python scripts/clean_datasets.py --dry-run
  python scripts/clean_datasets.py --no-backup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Trailing junk like: (غرفة وسائل تعليمية نجران …)
_TRAILING_NOISE_RE = re.compile(r"\s*\([^()]{8,}\)\s*$")
_HASH_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def strip_prompt_noise(prompt: str) -> str:
    prev = None
    text = prompt
    # Some rows may stack junk; strip repeatedly.
    while prev != text:
        prev = text
        text = _TRAILING_NOISE_RE.sub("", text).rstrip()
    return text


def serialize_answer(gt: Any) -> str:
    if gt is None:
        return ""
    if isinstance(gt, bool):
        return "true" if gt else "false"
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False, separators=(",", ": "))
    return str(gt).strip()


def normalize_coldstart_response(response: str) -> str:
    if not response:
        return response
    if "<answer>" in response and "</answer>" in response:
        return response.strip()
    match = _HASH_ANSWER_RE.search(response)
    if not match:
        return response.strip()
    answer_text = match.group(1).strip()
    body = _HASH_ANSWER_RE.sub("", response).rstrip()
    if not body.endswith("</think>"):
        # Keep structure consistent with reward_format expectations.
        if "<think>" in body and "</think>" not in body:
            body = body + "\n</think>"
    return f"{body}\n<answer>{answer_text}</answer>"


def clean_rlvr_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fixes: list[str] = []
    out = dict(row)
    meta = dict(out.get("metadata") or {})

    original = out.get("prompt", "")
    cleaned = strip_prompt_noise(original)
    if cleaned != original:
        out["prompt"] = cleaned
        fixes.append("strip_prompt_noise")

    gt = meta.get("ground_truth_answer", meta.get("ground_truth"))
    ans = serialize_answer(gt)
    if ans and str(out.get("answer") or "").strip() != ans:
        out["answer"] = ans
        fixes.append("fill_answer_from_metadata")

    # Keep metadata.ground_truth_answer as the canonical GT for the loader.
    if gt is not None and "ground_truth_answer" not in meta:
        meta["ground_truth_answer"] = gt
        fixes.append("normalize_gt_key")
    out["metadata"] = meta
    return out, fixes


def clean_coldstart_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fixes: list[str] = []
    out = dict(row)
    meta = dict(out.get("metadata") or {})

    original = out.get("prompt", "")
    cleaned = strip_prompt_noise(original)
    if cleaned != original:
        out["prompt"] = cleaned
        fixes.append("strip_prompt_noise")

    raw_resp = out.get("response") or ""
    new_resp = normalize_coldstart_response(raw_resp)
    if new_resp != raw_resp:
        out["response"] = new_resp
        fixes.append("normalize_answer_tags")

    # Prefer answer inside <answer>…</answer>, else metadata / existing field.
    m = re.search(r"<answer>(.*?)</answer>", out.get("response") or "", re.DOTALL)
    if m:
        extracted = m.group(1).strip()
        if str(out.get("answer") or "").strip() != extracted:
            out["answer"] = extracted
            fixes.append("sync_answer_from_response")
    else:
        gt = meta.get("ground_truth_answer", meta.get("ground_truth", out.get("answer")))
        ans = serialize_answer(gt)
        if ans and str(out.get("answer") or "").strip() != ans:
            out["answer"] = ans
            fixes.append("fill_answer_from_metadata")

    if "ground_truth_answer" not in meta:
        if out.get("answer") not in (None, ""):
            # Try to parse JSON answers back to objects for metadata consistency.
            raw_ans = out["answer"]
            try:
                meta["ground_truth_answer"] = json.loads(raw_ans)
            except (json.JSONDecodeError, TypeError):
                meta["ground_truth_answer"] = raw_ans
            fixes.append("normalize_gt_key")
    out["metadata"] = meta
    return out, fixes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_file(
    path: Path,
    kind: str,
    *,
    dry_run: bool,
) -> dict[str, int]:
    rows = _read_jsonl(path)
    cleaner = clean_rlvr_row if kind == "rlvr" else clean_coldstart_row
    cleaned_rows = []
    counts: dict[str, int] = {"rows": len(rows), "changed": 0}
    for row in rows:
        new_row, fixes = cleaner(row)
        cleaned_rows.append(new_row)
        if fixes:
            counts["changed"] += 1
            for fix in fixes:
                counts[fix] = counts.get(fix, 0) + 1

    # Post-clean audit
    if kind == "rlvr":
        counts["empty_answer"] = sum(1 for r in cleaned_rows if not str(r.get("answer") or "").strip())
        counts["prompt_noise_left"] = sum(
            1 for r in cleaned_rows if _TRAILING_NOISE_RE.search(r.get("prompt") or "")
        )
    else:
        counts["missing_answer_tag"] = sum(
            1
            for r in cleaned_rows
            if "<answer>" not in (r.get("response") or "") or "</answer>" not in (r.get("response") or "")
        )
        counts["prompt_noise_left"] = sum(
            1 for r in cleaned_rows if _TRAILING_NOISE_RE.search(r.get("prompt") or "")
        )

    if not dry_run:
        _write_jsonl(path, cleaned_rows)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean RLVR / coldstart JSONL datasets")
    parser.add_argument(
        "--rlvr",
        default=str(PROJECT_ROOT / "data" / "arabic_reasoning_rlvr.jsonl"),
    )
    parser.add_argument(
        "--coldstart",
        default=str(PROJECT_ROOT / "data" / "arabic_reasoning_coldstart.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    rlvr = Path(args.rlvr)
    coldstart = Path(args.coldstart)

    if not args.dry_run and not args.no_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = PROJECT_ROOT / "data" / f"backup_pre_clean_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rlvr, backup_dir / rlvr.name)
        shutil.copy2(coldstart, backup_dir / coldstart.name)
        print(f"Backup -> {backup_dir}")

    print("Cleaning RLVR…")
    rlvr_stats = clean_file(rlvr, "rlvr", dry_run=args.dry_run)
    print(json.dumps(rlvr_stats, ensure_ascii=False, indent=2))

    print("Cleaning coldstart…")
    cs_stats = clean_file(coldstart, "coldstart", dry_run=args.dry_run)
    print(json.dumps(cs_stats, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry-run only — no files written.")
    else:
        print("Done. Next: re-run SFT, then GRPO with log_completions=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
