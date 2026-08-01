#!/usr/bin/env python3
"""A/B teacher quality on the fixed verifier-first pipeline.

Generates the same N programmatic problems twice — once with each teacher model —
then scores both with the V4 acceptance gates (no metadata backfill).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator
from rlvr_synth.quality.acceptance import accept_sft_row, stamp_fresh_sft_audit


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _score_rows(rows: list[dict]) -> dict:
    accepted = 0
    reasons: Counter[str] = Counter()
    scored = []
    for row in rows:
        stamped = stamp_fresh_sft_audit(row)
        # Comparison runs may lack decontam metadata; score content gates only.
        ok, rs = accept_sft_row(stamped, require_decontam_clean=False)
        if ok:
            accepted += 1
        for r in rs:
            # Collapse schema noise prefixes
            key = r.split(":")[0] if ":" in r else r
            reasons[key] += 1
        scored.append(
            {
                "problem_id": stamped.get("problem_id"),
                "domain": stamped.get("domain"),
                "ok": ok,
                "reasons": rs,
                "prompt": (stamped.get("prompt") or "")[:160],
                "response_preview": (stamped.get("response") or "")[:400],
                "num_steps": (stamped.get("metadata") or {}).get("num_steps"),
                "think_tokens": (stamped.get("trace_audit") or {}).get("concision_tokens"),
            }
        )
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "accepted": accepted,
        "accept_rate": accepted / n,
        "reason_counts": dict(reasons.most_common()),
        "rows": scored,
    }


def _run_one(model: str, work_dir: Path, n: int, seed: int, budget_usd: float) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    cfg = SynthConfig(
        mode="pilot",
        n_families=n,
        domains=["gsm8k", "math", "math_comp", "logic"],
        traces_per_problem=1,
        seed=seed,
        work_dir=str(work_dir),
        backend="external",
        external_module=str(PACK_ROOT / "backends" / "dspy_backend.py"),
        external_config={
            "mode": "live",
            "model": model,
            "gepa_path": str(PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"),
            "budget_path": str(work_dir / "budget.json"),
            "budget_usd": budget_usd,
            "cache": False,
            "temperature": 0.35,
            "max_tokens": 3200,
        },
        max_alternate_methods=0,
        partitions={"sft_train": 1.0},
    )
    SynthOrchestrator(cfg).run(resume=False)
    return work_dir / "release_corpora" / "sft_train.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument(
        "--qwen-model",
        default="qwen/qwen3.7-flash",
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek/deepseek-v4-pro",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACK_ROOT / "outputs" / "ab_teacher_10",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: need OPENROUTER_API_KEY or DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    os.environ["RLVR_PACK_MODE"] = "live"
    os.environ["USE_OPENROUTER"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, model in (
        ("qwen37_flash", args.qwen_model),
        ("deepseek_v4_pro", args.deepseek_model),
    ):
        print(f"\n=== Generating n={args.n} with {label} ({model}) ===", flush=True)
        work = args.out_dir / label
        path = _run_one(model, work, args.n, args.seed, args.budget_usd)
        rows = _read_jsonl(path)
        scored = _score_rows(rows)
        results[label] = {
            "model": model,
            "path": str(path),
            "n_released": len(rows),
            "score": {k: v for k, v in scored.items() if k != "rows"},
            "rows": scored["rows"],
            "budget": json.loads((work / "budget.json").read_text(encoding="utf-8"))
            if (work / "budget.json").exists()
            else {},
        }
        print(
            json.dumps(
                {
                    "label": label,
                    "released": len(rows),
                    "accepted": scored["accepted"],
                    "accept_rate": scored["accept_rate"],
                    "top_rejects": scored["reason_counts"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    # Recommendation
    q = results["qwen37_flash"]["score"]
    d = results["deepseek_v4_pro"]["score"]
    if d["accept_rate"] > q["accept_rate"] + 0.05:
        pick = "deepseek/deepseek-v4-pro"
        why = "higher gate accept rate"
    elif q["accept_rate"] > d["accept_rate"] + 0.05:
        pick = "qwen/qwen3.7-flash"
        why = "higher gate accept rate"
    elif d["accepted"] >= q["accepted"]:
        pick = "deepseek/deepseek-v4-pro"
        why = "tie/near-tie — prefer Pro for CoT quality per GENERATION_PLAN"
    else:
        pick = "qwen/qwen3.7-flash"
        why = "tie broken toward higher accepts"

    summary = {
        "recommendation": pick,
        "reason": why,
        "qwen_accept_rate": q["accept_rate"],
        "deepseek_accept_rate": d["accept_rate"],
        "qwen_accepted": q["accepted"],
        "deepseek_accepted": d["accepted"],
        "n": args.n,
        "seed": args.seed,
    }
    out = {
        "summary": summary,
        "results": results,
    }
    out_path = args.out_dir / "AB_REPORT.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
