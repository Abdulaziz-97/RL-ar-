"""External role backend for this pack.

Two modes (set via config `mode` or env `RLVR_PACK_MODE`):

- **replay** — read frozen stage JSONL under `assets/replay_stages/` so the
  dial-20 reference can be reproduced offline (no API calls).
- **live** — generate problems with `vendor/synth/programmatic.py` and sample
  Formal Arabic teacher traces via GEPA-loaded `ArabicTeacher`.

The orchestrator talks only to the role protocols
(`LatentProblem` / `RenderedProblem` / `TraceCandidate`); this module adapts
them to the pack's DSPy + contracts stack.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))
if str(PACK_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACK_ROOT / "src"))
if str(PACK_ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(PACK_ROOT / "vendor"))

from rlvr_contracts.answer_spec import AnswerSpecError, infer_answer_spec_from_legacy, parse_answer_spec
from rlvr_contracts.response import parse_response
from rlvr_contracts.verifiers import verify_answer
from rlvr_synth.roles.protocols import LatentProblem, RenderedProblem, TraceCandidate
from rlvr_synth.roles.stub import StubArabicEditor


def _gt_to_spec(domain: str, gt: Any) -> dict[str, Any]:
    """Convert a programmatic ground-truth into a typed answer_spec dict."""
    reward_domain = "logic" if domain == "logic" else "math"
    try:
        return infer_answer_spec_from_legacy(gt, domain=reward_domain).to_dict()
    except AnswerSpecError:
        if isinstance(gt, (dict, list)):
            return parse_answer_spec({"type": "logic_json", "canonical": gt}).to_dict()
        raise


class ReplayProblemGenerator:
    """Replay latent families + Arabic renders from frozen stage JSONL."""

    def __init__(self, stages_dir: Path):
        self.by_seed: dict[tuple[str, int], dict] = {}
        self.renders_by_family: dict[str, dict] = {}
        for line in (stages_dir / "latent_spec.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            self.by_seed[row["domain"], int(row["seed"])] = row
        render_path = stages_dir / "arabic_render.jsonl"
        if render_path.exists():
            for line in render_path.open(encoding="utf-8"):
                row = json.loads(line)
                self.renders_by_family[row["family_id"]] = row

    def generate_family(self, domain: str, seed: int) -> LatentProblem:
        row = self.by_seed[domain, seed]
        return LatentProblem(
            family_id=row["family_id"],
            domain=row["domain"],
            answer_spec=row["answer_spec"],
            latent=row["latent"],
            partition=row.get("partition", "sft_train"),
        )

    def render_arabic(self, latent: LatentProblem, seed: int) -> RenderedProblem:
        cached = self.renders_by_family.get(latent.family_id)
        if cached:
            return RenderedProblem(
                problem_id=cached["problem_id"],
                family_id=cached["family_id"],
                domain=cached["domain"],
                partition=cached.get("partition", latent.partition),
                prompt=cached["prompt"],
                answer_spec=cached["answer_spec"],
                verifier_type=cached.get("verifier_type", "python"),
                verifier_version=cached.get("verifier_version", "rlvr-contracts-verifiers-v1"),
                provenance=cached.get("provenance") or {},
                licensing=cached.get("licensing") or {"license": "internal"},
                lineage=cached.get("lineage") or {},
                metadata=cached.get("metadata") or {},
            )
        prompt = str(latent.latent.get("prompt") or "")
        return RenderedProblem(
            problem_id=f"prob_{latent.domain}_{seed:05d}_live",
            family_id=latent.family_id,
            domain=latent.domain,
            partition=latent.partition,
            prompt=prompt,
            answer_spec=latent.answer_spec,
            provenance={"source": "live"},
            metadata={"ground_truth": latent.latent.get("ground_truth")},
        )


class ReplayTraceTeacher:
    """Return the frozen teacher response for a problem_id (repeated n times)."""

    def __init__(self, stages_dir: Path):
        self.by_pid: dict[str, str] = {}
        for name in ("multi_trace", "gates", "selected"):
            path = stages_dir / f"{name}.jsonl"
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row.get("problem_id") and row.get("response"):
                    self.by_pid[row["problem_id"]] = row["response"]

    def sample_traces(self, problem: RenderedProblem, n: int, seed: int) -> list[TraceCandidate]:
        response = self.by_pid.get(problem.problem_id)
        if response is None:
            raise KeyError(f"replay cache miss for {problem.problem_id}")
        parsed = parse_response(response)
        return [
            TraceCandidate(
                problem_id=problem.problem_id,
                response=response,
                method_id="dspy_0",
                teacher="replay",
                concision_tokens=len((parsed.think or "").split()),
                metadata={"seed": seed},
            )
            for _ in range(max(1, n))
        ]


class LiveProblemGenerator:
    """Build a latent family with the deterministic programmatic generators."""

    DOMAINS = ("gsm8k", "math", "math_comp", "logic")

    def __init__(self):
        from synth.programmatic import gen_gsm8k, gen_logic, gen_math, gen_math_comp
        import random

        self._random = random
        self._gens = {
            "gsm8k": gen_gsm8k,
            "math": gen_math,
            "math_comp": gen_math_comp,
            "logic": gen_logic,
        }

    def generate_family(self, domain: str, seed: int) -> LatentProblem:
        domain = domain if domain in self.DOMAINS else "math"
        rng = self._random.Random(seed ^ hash(domain) & 4294967295)
        sample = self._gens[domain](rng)
        return LatentProblem(
            family_id=f"fam_{sample.domain}_{seed:05d}",
            domain=sample.domain,
            answer_spec=_gt_to_spec(sample.domain, sample.ground_truth),
            latent={
                "kind": "programmatic",
                "prompt": sample.prompt,
                "ground_truth": sample.ground_truth,
                "num_steps": sample.num_steps,
                "solution_steps": sample.solution_steps,
                "meta_extra": sample.meta_extra,
                "seed": seed,
            },
        )

    def render_arabic(self, latent: LatentProblem, seed: int) -> RenderedProblem:
        prompt = str(latent.latent.get("prompt") or "")
        return RenderedProblem(
            problem_id=f"prob_{latent.domain}_{seed:05d}_{abs(hash(prompt)) % 10 ** 8:08d}",
            family_id=latent.family_id,
            domain=latent.domain,
            partition=latent.partition,
            prompt=prompt,
            answer_spec=latent.answer_spec,
            provenance={"source": "programmatic+dspy", "seed": seed},
            metadata={
                "ground_truth": latent.latent.get("ground_truth"),
                "num_steps": latent.latent.get("num_steps"),
            },
        )


class LiveTraceTeacher:
    """Call the GEPA ArabicTeacher (DeepSeek) to produce SFT-style <think>/<answer> traces."""

    def __init__(self, config: dict[str, Any]):
        from dotenv import load_dotenv
        import dspy
        from synth.dspy_teacher import BudgetState, load_teacher, make_deepseek_lm, ArabicTeacher

        load_dotenv(PACK_ROOT / ".env")
        model = str(config.get("model") or "deepseek-v4-flash")
        budget = BudgetState(
            Path(config.get("budget_path") or PACK_ROOT / "outputs" / "budget.json"),
            float(config.get("budget_usd") or 30.0),
        )
        lm = make_deepseek_lm(
            model=model if model != "deepseek-chat" else "deepseek-v4-flash",
            temperature=float(config.get("temperature") or 0.35),
            max_tokens=int(config.get("max_tokens") or 3200),
            budget=budget,
            cache=bool(config.get("cache", True)),
        )
        dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
        gepa = Path(config.get("gepa_path") or PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json")
        self.teacher = load_teacher(gepa) if gepa.exists() else ArabicTeacher()

    def sample_traces(self, problem: RenderedProblem, n: int, seed: int) -> list[TraceCandidate]:
        gt = problem.metadata.get("ground_truth")
        if gt is None:
            gt = problem.answer_spec.get("ground_truth_structured") or problem.answer_spec.get("canonical")
        gts = json.dumps(gt, ensure_ascii=False) if isinstance(gt, (dict, list)) else str(gt)
        out = []
        for i in range(max(1, n)):
            pred = self.teacher(domain=problem.domain, problem=problem.prompt, ground_truth=gts)
            response = (getattr(pred, "response", None) or "").strip() or "<think>\n\n</think>\n<answer></answer>"
            parsed = parse_response(response)
            out.append(
                TraceCandidate(
                    problem_id=problem.problem_id,
                    response=response,
                    method_id=f"dspy_{i}",
                    teacher="arabic_teacher_gepa_v2",
                    concision_tokens=len((parsed.think or "").split()),
                    metadata={"seed": seed + i},
                )
            )
        return out


class ContractVerifier:
    """Validate answer_spec + response format and score against ground truth."""

    def verify_problem(self, problem: RenderedProblem) -> tuple[bool, list[str]]:
        try:
            parse_answer_spec(problem.answer_spec)
        except Exception as exc:
            return (False, [f"answer_spec:{exc}"])
        if not problem.prompt.strip():
            return (False, ["empty_prompt"])
        return (True, [])

    def verify_trace(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        parsed = parse_response(trace.response)
        reasons = list(parsed.errors)
        if not parsed.format_ok:
            return (False, reasons or ["format_fail"])
        vr = verify_answer(trace.response, problem.answer_spec, from_completion=True)
        if not vr.ok:
            reasons.append(vr.reason)
            return (False, reasons)
        return (True, [])

    def verify_steps(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        import re

        ok, reasons = self.verify_trace(problem, trace)
        if not ok:
            return (False, reasons)
        think = parse_response(trace.response).think or ""
        lines = [ln for ln in think.splitlines() if ln.strip()]
        sentences = [s.strip() for s in re.split(r"[.。.!?\n]+", think) if s.strip()]
        if len(lines) < 2 and len(sentences) < 2 and (len(think.split()) < 20):
            return (False, ["insufficient_steps"])
        return (True, [])


class DSPyRoleBackend:
    """Factory that wires problem generator + teacher + verifier for the orchestrator."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        mode = str(cfg.get("mode") or os.environ.get("RLVR_PACK_MODE") or "replay")
        stages = Path(cfg.get("replay_stages") or PACK_ROOT / "assets" / "replay_stages")
        if mode == "replay":
            self.problem_generator = ReplayProblemGenerator(stages)
            self.trace_teacher = ReplayTraceTeacher(stages)
        else:
            self.problem_generator = LiveProblemGenerator()
            self.trace_teacher = LiveTraceTeacher(cfg)
        self.verifier = ContractVerifier()
        self.arabic_editor = StubArabicEditor()


def build_backend(config: dict | None = None) -> DSPyRoleBackend:
    """Entry point expected by `rlvr_synth.roles.external.ExternalModuleBackend`."""
    return DSPyRoleBackend(config)
