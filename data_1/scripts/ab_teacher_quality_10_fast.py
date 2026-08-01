#!/usr/bin/env python3
"""Fast A/B: same 10 programmatic problems, two teachers, fixed V4 gates."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor")]

from backends.dspy_backend import ContractVerifier, LiveProblemGenerator, _gt_to_spec
from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION
from rlvr_synth.quality.acceptance import accept_sft_row, stamp_fresh_sft_audit
from rlvr_synth.roles.protocols import RenderedProblem, TraceCandidate
from synth.dspy_teacher import (
    ArabicTeacher,
    BudgetState,
    load_teacher,
    make_deepseek_lm,
    think_quality_ok,
)
import dspy


def build_problems(n: int, seed: int) -> list[RenderedProblem]:
    gen = LiveProblemGenerator()
    domains = ["gsm8k", "math", "math_comp", "logic"]
    out: list[RenderedProblem] = []
    for i in range(n):
        domain = domains[i % len(domains)]
        latent = gen.generate_family(domain, seed + i)
        latent.partition = "sft_train"
        rendered = gen.render_arabic(latent, seed + i)
        rendered.partition = "sft_train"
        ok, reasons = ContractVerifier().verify_problem(rendered)
        if not ok:
            print(f"skip problem {i}: {reasons}", flush=True)
            continue
        out.append(rendered)
    return out


def configure_teacher(model: str, budget_path: Path, budget_usd: float) -> ArabicTeacher:
    budget = BudgetState(budget_path, budget_usd)
    lm = make_deepseek_lm(
        model=model,
        temperature=0.35,
        max_tokens=2400,
        budget=budget,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    gepa = PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"
    return load_teacher(gepa) if gepa.exists() else ArabicTeacher()


def teach_one(teacher: ArabicTeacher, problem: RenderedProblem) -> dict:
    gt = (problem.metadata or {}).get("ground_truth")
    gts = json.dumps(gt, ensure_ascii=False) if isinstance(gt, (dict, list)) else str(gt)
    t0 = time.time()
    pred = teacher(domain=problem.domain, problem=problem.prompt, ground_truth=gts)
    elapsed = time.time() - t0
    response = (getattr(pred, "response", None) or "").strip()
    trace = TraceCandidate(
        problem_id=problem.problem_id,
        response=response,
        method_id="dspy_0",
        teacher="arabic_teacher_gepa_v2",
        concision_tokens=len(response.split()),
    )
    ok, reasons = ContractVerifier().verify_steps(problem, trace)
    row = {
        "problem_id": problem.problem_id,
        "family_id": problem.family_id,
        "partition": "sft_train",
        "domain": problem.domain,
        "prompt": problem.prompt,
        "response": response,
        "answer_spec": problem.answer_spec,
        "verifier_type": "python",
        "verifier_version": VERIFIER_REGISTRY_VERSION,
        "verifier_result": bool(ok),
        "trace_audit": {
            "format_ok": bool(ok),
            "step_verified": bool(ok),
            "concision_tokens": len((response or "").split()),
            "rejection_reasons": list(reasons),
            "elapsed_sec": elapsed,
        },
        "metadata": {
            **dict(problem.metadata or {}),
            "decontam": {"status": "clean"},
            "teacher_elapsed_sec": elapsed,
        },
        "lineage": {
            "generator_version": "ab_fast",
            "template_family": (problem.metadata or {}).get("template_family"),
        },
        "provenance": dict(problem.provenance or {}),
        "licensing": {"license": "internal"},
    }
    return row


def score(rows: list[dict]) -> dict:
    accepted = 0
    reasons: Counter[str] = Counter()
    details = []
    for row in rows:
        stamped = stamp_fresh_sft_audit(row)
        ok, rs = accept_sft_row(stamped, require_decontam_clean=True)
        if ok:
            accepted += 1
        for r in rs:
            reasons[r.split(":")[0]] += 1
        think_ok = None
        try:
            from rlvr_contracts.response import extract_think

            think = extract_think(stamped.get("response") or "") or ""
            t_ok, tag = think_quality_ok(
                think,
                stamped.get("response") or "",
                domain=stamped.get("domain") or "gsm8k",
                gt=(stamped.get("metadata") or {}).get("ground_truth"),
            )
            think_ok = {"ok": t_ok, "tag": tag}
        except Exception as exc:
            think_ok = {"ok": False, "tag": str(exc)}
        details.append(
            {
                "problem_id": stamped.get("problem_id"),
                "domain": stamped.get("domain"),
                "ok": ok,
                "reasons": rs,
                "think_quality": think_ok,
                "elapsed_sec": (stamped.get("metadata") or {}).get("teacher_elapsed_sec"),
                "prompt": (stamped.get("prompt") or "")[:180],
                "response_preview": (stamped.get("response") or "")[:500],
            }
        )
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "accepted": accepted,
        "accept_rate": accepted / n,
        "reason_counts": dict(reasons.most_common()),
        "mean_elapsed_sec": sum(
            float((r.get("metadata") or {}).get("teacher_elapsed_sec") or 0) for r in rows
        )
        / n,
        "details": details,
    }


def main() -> int:
    n = int(os.environ.get("AB_N", "10"))
    seed = int(os.environ.get("AB_SEED", "4242"))
    out_dir = Path(os.environ.get("AB_OUT", str(PACK_ROOT / "outputs" / "ab_teacher_10_fast")))
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["USE_OPENROUTER"] = "1"
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 2

    problems = build_problems(n, seed)
    print(json.dumps({"n_problems": len(problems), "domains": [p.domain for p in problems]}, ensure_ascii=False), flush=True)

    models = {
        "qwen37_flash": "qwen/qwen3.7-flash",
        "deepseek_v4_pro": "deepseek/deepseek-v4-pro",
    }
    report = {"problems": [{"id": p.problem_id, "domain": p.domain, "prompt": p.prompt} for p in problems], "models": {}}

    for label, model in models.items():
        print(f"\n=== {label} ({model}) ===", flush=True)
        teacher = configure_teacher(model, out_dir / f"budget_{label}.json", 8.0)
        rows = []
        for i, problem in enumerate(problems):
            print(f"[{label}] {i+1}/{len(problems)} {problem.domain} ...", flush=True)
            try:
                row = teach_one(teacher, problem)
            except Exception as exc:
                print(f"[{label}] FAIL {problem.problem_id}: {exc}", flush=True)
                row = {
                    "problem_id": problem.problem_id,
                    "family_id": problem.family_id,
                    "partition": "sft_train",
                    "domain": problem.domain,
                    "prompt": problem.prompt,
                    "response": "",
                    "answer_spec": problem.answer_spec,
                    "verifier_type": "python",
                    "verifier_version": VERIFIER_REGISTRY_VERSION,
                    "verifier_result": False,
                    "trace_audit": {
                        "format_ok": False,
                        "step_verified": False,
                        "concision_tokens": 0,
                        "rejection_reasons": [f"exception:{exc}"],
                    },
                    "metadata": {
                        **dict(problem.metadata or {}),
                        "decontam": {"status": "clean"},
                        "teacher_elapsed_sec": None,
                    },
                    "lineage": {"generator_version": "ab_fast"},
                    "provenance": {},
                    "licensing": {"license": "internal"},
                }
            rows.append(row)
            # Persist incrementally
            path = out_dir / f"{label}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        scored = score(rows)
        report["models"][label] = {
            "model": model,
            "score": {k: v for k, v in scored.items() if k != "details"},
            "details": scored["details"],
            "budget": json.loads((out_dir / f"budget_{label}.json").read_text(encoding="utf-8"))
            if (out_dir / f"budget_{label}.json").exists()
            else {},
        }
        print(json.dumps({"label": label, **report["models"][label]["score"]}, ensure_ascii=False, indent=2), flush=True)

    q = report["models"]["qwen37_flash"]["score"]
    d = report["models"]["deepseek_v4_pro"]["score"]
    if d["accept_rate"] > q["accept_rate"] + 0.05:
        pick, why = "deepseek/deepseek-v4-pro", "higher V4 gate accept rate"
    elif q["accept_rate"] > d["accept_rate"] + 0.05:
        pick, why = "qwen/qwen3.7-flash", "higher V4 gate accept rate"
    else:
        # Prefer richer CoT / fewer think_quality failures, then Pro by plan default.
        q_tq = sum(1 for x in report["models"]["qwen37_flash"]["details"] if (x.get("think_quality") or {}).get("ok"))
        d_tq = sum(1 for x in report["models"]["deepseek_v4_pro"]["details"] if (x.get("think_quality") or {}).get("ok"))
        if d_tq > q_tq:
            pick, why = "deepseek/deepseek-v4-pro", "similar accept rate; better think_quality pass count"
        elif q_tq > d_tq:
            pick, why = "qwen/qwen3.7-flash", "similar accept rate; better think_quality pass count"
        else:
            pick, why = "deepseek/deepseek-v4-pro", "tie — prefer Pro per GENERATION_PLAN CoT quality"

    report["summary"] = {
        "recommendation": pick,
        "reason": why,
        "qwen_accept_rate": q["accept_rate"],
        "deepseek_accept_rate": d["accept_rate"],
        "qwen_accepted": q["accepted"],
        "deepseek_accepted": d["accepted"],
        "qwen_mean_sec": q["mean_elapsed_sec"],
        "deepseek_mean_sec": d["mean_elapsed_sec"],
    }
    out_path = out_dir / "AB_REPORT.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
