#!/usr/bin/env python3
"""Apply full answer-first Reverse-QA to an SFT jsonl (current-run post-pass).

For each row:
  1. Generate a new Arabic prompt from (GT, solution_steps) — not a paraphrase.
  2. Static gates + independent LLM re-solve must recover the same GT.
  3. Optionally re-teach CoT with DeepSeek so <think> matches the new prompt.
  4. On failure, keep the original prompt/response.

Usage:
  python data_1/scripts/apply_full_reverse_qa_sft.py \\
    --in path/to/sft.jsonl --out path/to/sft_rqa.jsonl --reteach --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parent
sys.path[:0] = [str(PACK), str(PACK / "src"), str(PACK / "vendor"), str(ROOT / "src")]

from rlvr_synth.roles.protocols import RenderedProblem
from rlvr_synth.roles.reverse_qa import (
    diversify_rendered_problem,
    make_openai_compatible_generate_fn,
    make_openai_compatible_solve_fn,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_to_rendered(row: dict[str, Any]) -> RenderedProblem:
    meta = dict(row.get("metadata") or {})
    return RenderedProblem(
        problem_id=str(row.get("problem_id") or ""),
        family_id=str(row.get("family_id") or ""),
        domain=str(row.get("domain") or "math"),
        partition=str(row.get("partition") or "sft_train"),
        prompt=str(row.get("prompt") or ""),
        answer_spec=dict(row.get("answer_spec") or {}),
        provenance={"seed": int(meta.get("seed") or row.get("seed") or 0)},
        metadata=meta,
    )


def _reteach(row: dict[str, Any], teacher) -> dict[str, Any]:
    rendered = _row_to_rendered(row)
    traces = teacher.sample_traces(rendered, 1, int((rendered.provenance or {}).get("seed") or 0))
    if not traces:
        return row
    tr = traces[0]
    out = dict(row)
    out["response"] = tr.response
    out["metadata"] = {
        **dict(row.get("metadata") or {}),
        "reverse_qa_reteach": True,
        "teacher": tr.teacher,
        "method_id": tr.method_id,
    }
    return out


def _process_one(
    row: dict[str, Any],
    *,
    generate_fn,
    solve_fn,
    teacher,
    reteach: bool,
) -> dict[str, Any]:
    rendered = _row_to_rendered(row)
    latent = {
        **dict(row.get("metadata") or {}),
        "solution_steps": (row.get("metadata") or {}).get("solution_steps")
        or (row.get("latent") or {}).get("solution_steps")
        or [],
        "template_family": (row.get("metadata") or {}).get("template_family"),
        "num_steps": (row.get("metadata") or {}).get("num_steps"),
        "meta_extra": (row.get("metadata") or {}).get("meta_extra") or {},
        "ground_truth": (row.get("answer_spec") or {}).get("canonical"),
    }
    diversified = diversify_rendered_problem(
        rendered,
        generate_fn=generate_fn,
        solve_fn=solve_fn,
        latent=latent,
        enabled=True,
        mode="full",
        require_resolve=True,
    )
    out = dict(row)
    rqa = (diversified.metadata or {}).get("reverse_qa") or {}
    if rqa.get("applied"):
        out["prompt"] = diversified.prompt
        out["problem_id"] = diversified.problem_id
        out["answer_spec"] = dict(row.get("answer_spec") or {})  # freeze
        out["metadata"] = {
            **dict(row.get("metadata") or {}),
            "reverse_qa": rqa,
            "oracle_ground_truth": (row.get("answer_spec") or {}).get("canonical"),
        }
        out["provenance"] = {
            **dict(row.get("provenance") or {}),
            **dict(diversified.provenance or {}),
        }
        if reteach and teacher is not None:
            out = _reteach(out, teacher)
    else:
        out["metadata"] = {
            **dict(row.get("metadata") or {}),
            "reverse_qa": rqa,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reteach", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("REVERSE_QA_WORKERS", "16")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default=os.environ.get("REVERSE_QA_MODEL", "deepseek-v4-flash"))
    args = ap.parse_args()

    rows = _read_jsonl(args.inp)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    generate_fn = make_openai_compatible_generate_fn(model=args.model)
    solve_fn = make_openai_compatible_solve_fn(model=args.model)
    teacher = None
    if args.reteach:
        from backends.dspy_backend import LiveTraceTeacher

        # Default reteach to the same Flash model as Reverse-QA generate/resolve
        # so the post-pass stays ~3 Flash calls/sample (gen+resolve+reteach), not Pro.
        reteach_model = (
            os.environ.get("REVERSE_QA_RETEACH_MODEL")
            or os.environ.get("REVERSE_QA_MODEL")
            or args.model
            or "deepseek-v4-flash"
        )
        teacher = LiveTraceTeacher(
            {
                "model": reteach_model,
                "budget_path": str(args.out.with_suffix(".rqa_budget.json")),
                "budget_usd": float(os.environ.get("SFT_BUDGET_USD", "200")),
                "max_tokens": int(os.environ.get("TEACHER_MAX_TOKENS", "8192")),
            }
        )

    out_rows: list[dict[str, Any] | None] = [None] * len(rows)
    applied = 0
    failed = 0

    def work(idx_row: tuple[int, dict[str, Any]]):
        i, row = idx_row
        return i, _process_one(
            row,
            generate_fn=generate_fn,
            solve_fn=solve_fn,
            teacher=teacher,
            reteach=args.reteach,
        )

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, (i, r)) for i, r in enumerate(rows)]
        for fut in as_completed(futs):
            i, row = fut.result()
            out_rows[i] = row
            rqa = (row.get("metadata") or {}).get("reverse_qa") or {}
            if rqa.get("applied"):
                applied += 1
            else:
                failed += 1
            done = applied + failed
            if done == 1 or done % 25 == 0 or done == len(rows):
                print(
                    f"[reverse_qa_full] {done}/{len(rows)} applied={applied} fallback={failed}",
                    flush=True,
                )

    final = [r for r in out_rows if r is not None]
    _write_jsonl(args.out, final)
    summary = {
        "input": str(args.inp),
        "output": str(args.out),
        "n": len(final),
        "applied": applied,
        "fallback": failed,
        "reteach": bool(args.reteach),
        "mode": "full",
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    args.out.with_suffix(".rqa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
