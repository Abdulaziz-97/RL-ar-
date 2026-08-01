#!/usr/bin/env python3
"""Generate a verified IFEval-like SFT sidecar (no MCQ / AraPro data).

Rows use answer_spec.type=constraint_set and a deterministic teacher body that
passes the structured constraint verifier. No API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor")]

from rlvr_contracts.verifiers import verify_answer  # noqa: E402
from synth.programmatic import gen_ifeval_multiconstraint  # noqa: E402


def _read_prompts(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    out.add(re.sub(r"\s+", " ", line.strip()))
                    continue
                prompt = row.get("prompt") or row.get("question") or ""
                if prompt:
                    out.add(re.sub(r"\s+", " ", str(prompt).strip()))
    return out


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_row(sample, *, seed: int, idx: int) -> dict[str, Any]:
    meta = dict(sample.meta_extra or {})
    answer_spec = dict(meta.pop("answer_spec"))
    body = str(meta.pop("teacher_body"))
    think = "\n".join(sample.solution_steps)
    # Keep natural constrained body in <answer> so mixed SFT still sees format cues.
    response = f"<think>\n{think}\n</think>\n<answer>\n{body}\n</answer>"
    family_id = f"fam_ifeval_sidecar_{idx:05d}_s{seed}"
    problem_id = f"prob_ifeval_sidecar_{idx:05d}_{_prompt_hash(sample.prompt)[:8]}"
    return {
        "problem_id": problem_id,
        "family_id": family_id,
        "partition": "sft_train",
        "domain": "ifeval_multiconstraint",
        "prompt": sample.prompt,
        "answer_spec": answer_spec,
        "verifier_type": "python",
        "verifier_version": "rlvr-contracts-verifiers-v1",
        "provenance": {"source": "programmatic_ifeval_sidecar", "seed": seed + idx},
        "licensing": {"license": "internal"},
        "lineage": {"generator_version": "rlvr_synth_ifeval_v5"},
        "metadata": {
            **meta,
            "num_steps": sample.num_steps,
            "seed": seed + idx,
            "sidecar": "ifeval",
            "decontam": {"status": "clean", "reasons": [], "mode": "sidecar_exact_prompt"},
        },
        "response": response,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4400)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="JSONL prompts to avoid exact-duplicating (repeatable)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = _read_prompts(list(args.reference))
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = args.n * 80

    while len(rows) < args.n and attempts < max_attempts:
        attempts += 1
        sample = gen_ifeval_multiconstraint(rng)
        key = re.sub(r"\s+", " ", sample.prompt.strip())
        if key in seen:
            continue
        row = _build_row(sample, seed=args.seed, idx=len(rows) + 1)
        result = verify_answer(row["response"], row["answer_spec"], from_completion=True)
        if not result.ok:
            continue
        ar = sum(1 for c in row["prompt"] + row["response"] if "\u0600" <= c <= "\u06ff")
        if ar < 40:
            continue
        seen.add(key)
        rows.append(row)

    if len(rows) < args.n:
        raise SystemExit(
            f"only generated {len(rows)}/{args.n} verified IFEval rows after {attempts} attempts"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "rows": len(rows),
        "seed": args.seed,
        "domain": "ifeval_multiconstraint",
        "answer_type": "constraint_set",
        "mcq_generated": False,
        "out": str(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
