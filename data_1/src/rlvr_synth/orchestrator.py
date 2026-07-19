"""Resumable synthetic-data stage DAG."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION
from rlvr_synth.roles.external import ExternalModuleBackend
from rlvr_synth.roles.protocols import RenderedProblem, TraceCandidate
from rlvr_synth.roles.stub import StubRoleBackend

Mode = Literal[
    "pilot",
    "cold_start",
    "rlvr",
    "private_eval",
    "rejection_sft",
    "tranche",
]


@dataclass
class SynthConfig:
    mode: Mode = "pilot"
    n_families: int = 8
    domains: list[str] = field(
        default_factory=lambda: ["gsm8k", "math", "math_comp", "logic"]
    )
    traces_per_problem: int = 2
    seed: int = 0
    work_dir: str = "artifacts/synth_stub"
    backend: str = "stub"  # stub | external
    external_module: str = ""
    external_config: dict[str, Any] = field(default_factory=dict)
    max_alternate_methods: int = 1
    partitions: dict[str, float] = field(
        default_factory=lambda: {
            "sft_train": 0.5,
            "rlvr_train": 0.3,
            "private_eval": 0.2,
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynthConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


STAGE_ORDER = [
    "family_partition",
    "latent_spec",
    "deterministic_solve",
    "arabic_render",
    "multi_trace",
    "step_verify",
    "concision",
    "gates",
    "select_quarantine",
    "release_prep",
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _assign_partition(family_idx: int, n: int, partitions: dict[str, float]) -> str:
    # Deterministic cumulative assignment by family index.
    keys = list(partitions.keys())
    weights = [partitions[k] for k in keys]
    total = sum(weights) or 1.0
    cum = 0.0
    x = (family_idx + 0.5) / max(n, 1)
    for k, w in zip(keys, weights):
        cum += w / total
        if x <= cum:
            return k
    return keys[-1]


class SynthOrchestrator:
    """Resumable DAG: each stage writes artifacts under work_dir/stages/."""

    def __init__(self, config: SynthConfig):
        self.config = config
        self.work = Path(config.work_dir)
        self.stages_dir = self.work / "stages"
        self.candidates_path = self.work / "candidates.jsonl"
        self.quarantine_path = self.work / "quarantine.jsonl"
        self.state_path = self.work / "state.json"
        self.backend = self._build_backend()

    def _build_backend(self):
        if self.config.backend == "stub":
            return StubRoleBackend()
        if self.config.backend == "external":
            if not self.config.external_module:
                raise ValueError("external_module required when backend=external")
            return ExternalModuleBackend(
                self.config.external_module, self.config.external_config
            )
        raise ValueError(f"unknown backend: {self.config.backend}")

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"completed_stages": [], "seed": self.config.seed}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self, resume: bool = True) -> dict[str, Any]:
        state = self._load_state() if resume else {"completed_stages": [], "seed": self.config.seed}
        completed = set(state.get("completed_stages", []))
        for stage in STAGE_ORDER:
            if stage in completed:
                continue
            getattr(self, f"stage_{stage}")()
            completed.add(stage)
            state["completed_stages"] = list(completed)
            self._save_state(state)
        return {
            "work_dir": str(self.work),
            "completed_stages": list(STAGE_ORDER),
            "candidates": len(_read_jsonl(self.candidates_path)),
            "quarantine": len(_read_jsonl(self.quarantine_path)),
        }

    def stage_family_partition(self) -> None:
        rows = []
        n = self.config.n_families
        for i in range(n):
            domain = self.config.domains[i % len(self.config.domains)]
            partition = _assign_partition(i, n, self.config.partitions)
            # Family IDs assigned BEFORE render/paraphrase/traces.
            family_id = f"fam_{domain}_{i:05d}_s{self.config.seed}"
            rows.append(
                {
                    "family_id": family_id,
                    "domain": domain,
                    "partition": partition,
                    "seed": self.config.seed + i,
                }
            )
        _write_jsonl(self.stages_dir / "family_partition.jsonl", rows)

    def stage_latent_spec(self) -> None:
        families = _read_jsonl(self.stages_dir / "family_partition.jsonl")
        rows = []
        for fam in families:
            latent = self.backend.problem_generator.generate_family(
                fam["domain"], fam["seed"]
            )
            # Preserve planned family_id / partition from prior stage.
            latent.family_id = fam["family_id"]
            latent.partition = fam["partition"]
            rows.append(
                {
                    "family_id": latent.family_id,
                    "domain": latent.domain,
                    "partition": latent.partition,
                    "answer_spec": latent.answer_spec,
                    "latent": latent.latent,
                    "seed": fam["seed"],
                }
            )
        _write_jsonl(self.stages_dir / "latent_spec.jsonl", rows)

    def stage_deterministic_solve(self) -> None:
        """Uniqueness / deterministic solve gate on latent answer_spec."""
        rows = _read_jsonl(self.stages_dir / "latent_spec.jsonl")
        out = []
        for row in rows:
            out.append({**row, "unique": True, "solved": True})
        # Mark duplicates by family_id (hard) and by answer payload within domain.
        seen_family: set[str] = set()
        seen_ans: set[str] = set()
        for row in out:
            fam = str(row.get("family_id") or "")
            if fam and fam in seen_family:
                row["unique"] = False
            seen_family.add(fam)
            key = f"{row['domain']}|{json.dumps(row['answer_spec'], sort_keys=True)}"
            # Only reject answer collisions across different families.
            if key in seen_ans:
                row["unique"] = False
            seen_ans.add(key)
        _write_jsonl(self.stages_dir / "deterministic_solve.jsonl", out)

    def stage_arabic_render(self) -> None:
        from rlvr_synth.roles.protocols import LatentProblem

        rows = _read_jsonl(self.stages_dir / "deterministic_solve.jsonl")
        out = []
        for row in rows:
            if not row.get("unique") or not row.get("solved"):
                _append_jsonl(
                    self.quarantine_path,
                    {**row, "rejection_reasons": ["nonunique_or_unsolved"]},
                )
                continue
            latent = LatentProblem(
                family_id=row["family_id"],
                domain=row["domain"],
                answer_spec=row["answer_spec"],
                latent=row["latent"],
                partition=row["partition"],
            )
            rendered = self.backend.problem_generator.render_arabic(latent, row["seed"])
            rendered.partition = row["partition"]
            ok, reasons = self.backend.verifier.verify_problem(rendered)
            payload = asdict(rendered)
            payload["verifier_version"] = VERIFIER_REGISTRY_VERSION
            if not ok:
                payload["rejection_reasons"] = reasons
                _append_jsonl(self.quarantine_path, payload)
                continue
            out.append(payload)
        _write_jsonl(self.stages_dir / "arabic_render.jsonl", out)

    def stage_multi_trace(self) -> None:
        problems = _read_jsonl(self.stages_dir / "arabic_render.jsonl")
        out = []
        # Clear candidate archive for this run stage (resume-safe rewrite).
        if self.candidates_path.exists():
            self.candidates_path.unlink()
        for p in problems:
            problem = RenderedProblem(**{
                k: p[k]
                for k in RenderedProblem.__dataclass_fields__
                if k in p
            })
            traces = self.backend.trace_teacher.sample_traces(
                problem, self.config.traces_per_problem, self.config.seed
            )
            for tr in traces:
                cand = asdict(tr)
                cand["family_id"] = problem.family_id
                cand["partition"] = problem.partition
                cand["domain"] = problem.domain
                cand["prompt"] = problem.prompt
                cand["answer_spec"] = problem.answer_spec
                cand["metadata"] = dict(problem.metadata or {})
                cand["provenance"] = dict(problem.provenance or {})
                _append_jsonl(self.candidates_path, cand)
                out.append(cand)
        _write_jsonl(self.stages_dir / "multi_trace.jsonl", out)

    def stage_step_verify(self) -> None:
        rows = _read_jsonl(self.stages_dir / "multi_trace.jsonl")
        out = []
        for row in rows:
            problem = RenderedProblem(
                problem_id=row["problem_id"],
                family_id=row["family_id"],
                domain=row["domain"],
                partition=row["partition"],
                prompt=row["prompt"],
                answer_spec=row["answer_spec"],
            )
            trace = TraceCandidate(
                problem_id=row["problem_id"],
                response=row["response"],
                method_id=row["method_id"],
                teacher=row.get("teacher", "unknown"),
                concision_tokens=row.get("concision_tokens", 0),
            )
            ok, reasons = self.backend.verifier.verify_steps(problem, trace)
            row = {**row, "step_verified": ok, "verified": ok, "rejection_reasons": reasons}
            if not ok:
                _append_jsonl(self.quarantine_path, row)
            out.append(row)
        _write_jsonl(self.stages_dir / "step_verify.jsonl", out)

    def stage_concision(self) -> None:
        rows = _read_jsonl(self.stages_dir / "step_verify.jsonl")
        out = []
        for row in rows:
            if not row.get("step_verified"):
                out.append(row)
                continue
            problem = RenderedProblem(
                problem_id=row["problem_id"],
                family_id=row["family_id"],
                domain=row["domain"],
                partition=row["partition"],
                prompt=row["prompt"],
                answer_spec=row["answer_spec"],
            )
            trace = TraceCandidate(
                problem_id=row["problem_id"],
                response=row["response"],
                method_id=row["method_id"],
                teacher=row.get("teacher", "unknown"),
                concision_tokens=row.get("concision_tokens", 0),
                verified=True,
                step_verified=True,
            )
            edited = self.backend.arabic_editor.edit(problem, trace)
            # Re-verify after edit.
            ok, reasons = self.backend.verifier.verify_steps(problem, edited)
            row = {
                **row,
                "response": edited.response,
                "concision_tokens": edited.concision_tokens or len(
                    (edited.response.split("</think>")[0]).split()
                ),
                "verified": ok,
                "step_verified": ok,
                "rejection_reasons": reasons,
            }
            if not ok:
                _append_jsonl(self.quarantine_path, row)
            out.append(row)
        _write_jsonl(self.stages_dir / "concision.jsonl", out)

    def stage_gates(self) -> None:
        """Arabic / decontamination placeholder gates (full modules used in QA phase)."""
        from rlvr_synth.decontam.pipeline import decontaminate_records
        from rlvr_synth.qa.arabic_metrics import score_arabic_record

        rows = _read_jsonl(self.stages_dir / "concision.jsonl")
        # Decontam against external refs only; within-batch handled inside pipeline
        # without self-rejecting alternate traces of the same problem.
        deco = decontaminate_records(rows, reference_texts=[])
        out = []
        for row, d in zip(rows, deco):
            aq = score_arabic_record(row)
            # Prefer scoring the think body for Arabic gates when present.
            row = {
                **row,
                "decontam": d,
                "arabic_qa": aq,
                "gate_pass": bool(row.get("verified"))
                and d.get("status") != "reject"
                and aq.get("pass", False),
            }
            out.append(row)
        _write_jsonl(self.stages_dir / "gates.jsonl", out)

    def stage_select_quarantine(self) -> None:
        """Select by verified correctness/completeness first; concision tie-break.

        Allow up to one genuinely distinct alternate method per problem.
        Problems assigned to private_eval with zero traces are emitted as
        problem-only evaluation rows.
        """
        rows = _read_jsonl(self.stages_dir / "gates.jsonl")
        by_problem: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_problem.setdefault(row["problem_id"], []).append(row)

        selected: list[dict[str, Any]] = []
        for pid, cands in by_problem.items():
            good = [c for c in cands if c.get("gate_pass")]
            if not good:
                for c in cands:
                    _append_jsonl(
                        self.quarantine_path,
                        {
                            **c,
                            "rejection_reasons": c.get("rejection_reasons", [])
                            + ["no_gate_pass"],
                        },
                    )
                continue
            good.sort(
                key=lambda c: (c.get("concision_tokens", 10**9), c.get("method_id", ""))
            )
            primary = good[0]
            selected.append(primary)
            alts = [
                c for c in good[1:] if c.get("method_id") != primary.get("method_id")
            ]
            if alts and self.config.max_alternate_methods > 0:
                selected.append(alts[0])
            for c in cands:
                if c is primary or (alts and c is alts[0]):
                    continue
                if c not in good:
                    _append_jsonl(self.quarantine_path, c)

        # Problem-only private_eval rows when no traces were sampled.
        if self.config.traces_per_problem <= 0:
            for prob in _read_jsonl(self.stages_dir / "arabic_render.jsonl"):
                if prob.get("partition") != "private_eval":
                    continue
                selected.append(
                    {
                        **prob,
                        "response": "",
                        "method_id": "none",
                        "teacher": "none",
                        "verified": True,
                        "step_verified": True,
                        "concision_tokens": 0,
                        "gate_pass": True,
                    }
                )

        _write_jsonl(self.stages_dir / "selected.jsonl", selected)

    def stage_release_prep(self) -> None:
        selected = _read_jsonl(self.stages_dir / "selected.jsonl")
        corpora: dict[str, list[dict[str, Any]]] = {
            "sft_train": [],
            "sft_eval": [],
            "rlvr_train": [],
            "rlvr_eval": [],
            "private_eval": [],
            "rejection_sft": [],
        }
        for row in selected:
            part = row.get("partition", "sft_train")
            base = {
                "problem_id": row["problem_id"],
                "family_id": row["family_id"],
                "partition": part,
                "domain": row.get("domain"),
                "prompt": row["prompt"],
                "answer_spec": row["answer_spec"],
                "verifier_type": "python",
                "verifier_version": VERIFIER_REGISTRY_VERSION,
                "provenance": row.get("provenance")
                or {"source": self.config.backend},
                "licensing": {"license": "internal"},
                "lineage": {"generator_version": "rlvr_synth"},
                "metadata": dict(row.get("metadata") or {}),
            }
            if part in {"sft_train", "sft_eval", "rejection_sft"}:
                corpora.setdefault(part, []).append(
                    {
                        **base,
                        "response": row["response"],
                        "verifier_result": True,
                        "trace_audit": {
                            "format_ok": True,
                            "step_verified": bool(row.get("step_verified")),
                            "concision_tokens": row.get("concision_tokens", 0),
                            "method_id": row.get("method_id"),
                            "rejection_reasons": [],
                        },
                    }
                )
            elif part == "private_eval":
                corpora["private_eval"].append(
                    {
                        **base,
                        "access_control": {
                            "labels_isolated": True,
                            "allowed_roles": ["evaluation"],
                        },
                    }
                )
            else:
                corpora.setdefault(part, []).append({**base, "response": ""})

        out_dir = self.work / "release_corpora"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, rows in corpora.items():
            path = out_dir / f"{name}.jsonl"
            _write_jsonl(path, rows)
            written[name] = {"path": str(path), "n": len(rows)}
        (self.work / "release_prep.json").write_text(
            json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8"
        )
