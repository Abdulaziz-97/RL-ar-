"""Resumable synthetic-data stage DAG."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from rlvr_contracts.response import extract_think
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
    decontam_reference_paths: list[str] = field(default_factory=list)
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

    def validate(self) -> None:
        if self.n_families < 0:
            raise ValueError("n_families must be non-negative")
        if self.traces_per_problem < 0:
            raise ValueError("traces_per_problem must be non-negative")
        if self.max_alternate_methods < 0:
            raise ValueError("max_alternate_methods must be non-negative")
        if self.n_families and not self.domains:
            raise ValueError("domains must not be empty when generating families")
        if not self.partitions:
            raise ValueError("partitions must not be empty")
        weights = list(self.partitions.values())
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in weights
        ):
            raise ValueError("partition weights must be finite non-negative numbers")
        if sum(float(weight) for weight in weights) <= 0:
            raise ValueError("at least one partition weight must be positive")


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

# These partitions contain problems/labels, not teacher demonstrations.
PROMPT_ONLY_PARTITIONS = {"rlvr_train", "rlvr_eval", "private_eval"}


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
        config.validate()
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

    def _config_fingerprint(self) -> str:
        # Exclude throughput/path knobs so raising TEACHER_WORKERS mid-run can resume.
        payload_obj = asdict(self.config)
        ext = dict(payload_obj.get("external_config") or {})
        for volatile in (
            "teacher_workers",
            "budget_path",
            "teacher_retries",
            "cache",
        ):
            ext.pop(volatile, None)
        payload_obj["external_config"] = ext
        payload = json.dumps(payload_obj, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _reset_managed_outputs(self) -> None:
        managed = [
            self.state_path,
            self.candidates_path,
            self.quarantine_path,
            self.work / "release_prep.json",
        ]
        managed.extend(self.stages_dir / f"{stage}.jsonl" for stage in STAGE_ORDER)
        managed.extend(
            self.work / "release_corpora" / f"{name}.jsonl"
            for name in (
                "sft_train",
                "sft_eval",
                "rlvr_train",
                "rlvr_eval",
                "private_eval",
                "rejection_sft",
            )
        )
        for path in managed:
            if path.exists():
                path.unlink()

    def run(self, resume: bool = True) -> dict[str, Any]:
        fingerprint = self._config_fingerprint()
        if resume:
            state = self._load_state()
            previous = state.get("config_fingerprint")
            if previous is None and state.get("completed_stages"):
                raise RuntimeError(
                    "refusing to resume legacy state without a configuration "
                    "fingerprint; rerun with resume=False/--no-resume"
                )
            if previous and previous != fingerprint:
                raise RuntimeError(
                    "refusing to resume with a changed configuration; "
                    "rerun with resume=False/--no-resume"
                )
        else:
            self._reset_managed_outputs()
            state = {"completed_stages": [], "seed": self.config.seed}
        state["config_fingerprint"] = fingerprint
        completed = set(state.get("completed_stages", []))
        for stage in STAGE_ORDER:
            if stage in completed:
                continue
            getattr(self, f"stage_{stage}")()
            completed.add(stage)
            state["completed_stages"] = [
                name for name in STAGE_ORDER if name in completed
            ]
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
            # Prefer content-derived family_id from the generator when present.
            if not getattr(latent, "family_id", None):
                latent.family_id = fam["family_id"]
            latent.partition = fam["partition"]
            rows.append(
                {
                    "family_id": latent.family_id,
                    "planned_family_id": fam["family_id"],
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
        # Mark duplicate families and duplicate generated prompts. Different
        # problems are allowed to share an answer (e.g. many problems equal 0).
        seen_family: set[str] = set()
        seen_problem: set[str] = set()
        for row in out:
            fam = str(row.get("family_id") or "")
            if fam and fam in seen_family:
                row["unique"] = False
            seen_family.add(fam)
            latent = row.get("latent") or {}
            prompt = " ".join(str(latent.get("prompt") or "").split()).casefold()
            key = f"{row['domain']}|{prompt}"
            if prompt and key in seen_problem:
                row["unique"] = False
            if prompt:
                seen_problem.add(key)
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

    def _teacher_workers(self) -> int:
        """Frontier-lab throughput: parallel teacher calls across problems."""
        ext = self.config.external_config or {}
        raw = ext.get("teacher_workers")
        if raw is None:
            raw = os.environ.get("TEACHER_WORKERS", "1")
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 1
        return max(1, n)

    def _pack_trace_candidates(
        self, problem: RenderedProblem, traces: list[TraceCandidate]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tr in traces:
            cand = asdict(tr)
            cand["family_id"] = problem.family_id
            cand["partition"] = problem.partition
            cand["domain"] = problem.domain
            cand["prompt"] = problem.prompt
            cand["answer_spec"] = problem.answer_spec
            cand["metadata"] = {
                **dict(problem.metadata or {}),
                **dict(tr.metadata or {}),
            }
            cand["provenance"] = dict(problem.provenance or {})
            rows.append(cand)
        return rows

    def stage_multi_trace(self) -> None:
        problems = _read_jsonl(self.stages_dir / "arabic_render.jsonl")
        partial_path = self.stages_dir / "multi_trace.partial.jsonl"
        resume_teacher = os.environ.get("DATAGEN_RESUME_MULTI_TRACE", "1") == "1"

        # Load prior per-problem progress (crash-safe); never skip gates later.
        done_by_pid: dict[str, list[dict[str, Any]]] = {}
        if resume_teacher and partial_path.exists():
            for row in _read_jsonl(partial_path):
                pid = str(row.get("problem_id") or "")
                if pid:
                    done_by_pid.setdefault(pid, []).append(row)
            print(
                f"[multi_trace] resume partial problems={len(done_by_pid)} from {partial_path}",
                flush=True,
            )
        elif self.candidates_path.exists():
            self.candidates_path.unlink()
        if not resume_teacher and partial_path.exists():
            partial_path.unlink()

        prompt_only: list[tuple[int, RenderedProblem]] = []
        need_teacher: list[tuple[int, RenderedProblem]] = []
        for idx, p in enumerate(problems):
            problem = RenderedProblem(
                **{k: p[k] for k in RenderedProblem.__dataclass_fields__ if k in p}
            )
            if problem.partition in PROMPT_ONLY_PARTITIONS:
                prompt_only.append((idx, problem))
            else:
                need_teacher.append((idx, problem))

        by_idx: dict[int, list[dict[str, Any]]] = {}
        for idx, problem in prompt_only:
            if problem.problem_id in done_by_pid:
                by_idx[idx] = done_by_pid[problem.problem_id]
                continue
            traces = [
                TraceCandidate(
                    problem_id=problem.problem_id,
                    response="",
                    method_id="none",
                    teacher="none",
                    concision_tokens=0,
                )
            ]
            by_idx[idx] = self._pack_trace_candidates(problem, traces)

        pending_teacher = [
            item for item in need_teacher if item[1].problem_id not in done_by_pid
        ]
        for idx, problem in need_teacher:
            if problem.problem_id in done_by_pid:
                by_idx[idx] = done_by_pid[problem.problem_id]

        workers = self._teacher_workers() if pending_teacher else 1
        write_lock = threading.Lock()
        done = len(need_teacher) - len(pending_teacher)
        t0 = time.perf_counter()
        if pending_teacher:
            print(
                f"[multi_trace] teacher pending={len(pending_teacher)} "
                f"already={done} workers={workers} "
                f"traces_per={self.config.traces_per_problem}",
                flush=True,
            )

        def _teach_one(item: tuple[int, RenderedProblem]) -> tuple[int, list[dict[str, Any]]]:
            idx, problem = item
            traces = self.backend.trace_teacher.sample_traces(
                problem, self.config.traces_per_problem, self.config.seed
            )
            return idx, self._pack_trace_candidates(problem, traces)

        def _persist_partial(rows: list[dict[str, Any]]) -> None:
            with write_lock:
                with open(partial_path, "a", encoding="utf-8") as f:
                    for cand in rows:
                        f.write(json.dumps(cand, ensure_ascii=False) + "\n")

        if pending_teacher and workers <= 1:
            for item in pending_teacher:
                idx, rows = _teach_one(item)
                by_idx[idx] = rows
                _persist_partial(rows)
                done += 1
                if done == 1 or done % 25 == 0 or done == len(need_teacher):
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(
                        f"[multi_trace] {done}/{len(need_teacher)} "
                        f"({done / elapsed:.2f} problems/s)",
                        flush=True,
                    )
        elif pending_teacher:
            # Batch submissions so thread-local HTTP/LM clients can be GC'd between
            # pools. Submitting all ~5k futures at once with 96 workers blows soft
            # nofile=1024 (EMFILE on budget.json / sockets).
            batch_size = max(workers * 4, workers)
            for batch_start in range(0, len(pending_teacher), batch_size):
                batch = pending_teacher[batch_start : batch_start + batch_size]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_teach_one, item) for item in batch]
                    for fut in as_completed(futures):
                        idx, rows = fut.result()
                        by_idx[idx] = rows
                        _persist_partial(rows)
                        done += 1
                        if done == 1 or done % 25 == 0 or done == len(need_teacher):
                            elapsed = max(time.perf_counter() - t0, 1e-6)
                            print(
                                f"[multi_trace] {done}/{len(need_teacher)} "
                                f"({done / elapsed:.2f} problems/s)",
                                flush=True,
                            )
                # Best-effort budget flush + allow sockets/FDs to settle between batches.
                budget = getattr(getattr(self.backend, "trace_teacher", None), "_budget", None)
                if budget is not None and hasattr(budget, "flush"):
                    try:
                        budget.flush()
                    except Exception:
                        pass

        # Flush teacher budget buffer if present.
        teacher = getattr(self.backend, "trace_teacher", None)
        budget = getattr(teacher, "_budget", None)
        if budget is not None and hasattr(budget, "flush"):
            try:
                budget.flush()
            except Exception:
                pass

        out: list[dict[str, Any]] = []
        if self.candidates_path.exists():
            self.candidates_path.unlink()
        for idx in range(len(problems)):
            rows = by_idx.get(idx) or []
            for cand in rows:
                _append_jsonl(self.candidates_path, cand)
                out.append(cand)
        _write_jsonl(self.stages_dir / "multi_trace.jsonl", out)
        if partial_path.exists():
            partial_path.unlink()
        if need_teacher:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            print(
                f"[multi_trace] done taught={len(need_teacher)} "
                f"prompt_only={len(prompt_only)} in {elapsed:.1f}s "
                f"({len(need_teacher) / elapsed:.2f} problems/s)",
                flush=True,
            )

    def stage_step_verify(self) -> None:
        rows = _read_jsonl(self.stages_dir / "multi_trace.jsonl")
        workers = max(1, int(os.environ.get("VERIFY_WORKERS", "16")))
        write_lock = threading.Lock()

        def _verify_one(row: dict[str, Any]) -> dict[str, Any]:
            problem = RenderedProblem(
                problem_id=row["problem_id"],
                family_id=row["family_id"],
                domain=row["domain"],
                partition=row["partition"],
                prompt=row["prompt"],
                answer_spec=row["answer_spec"],
                metadata=dict(row.get("metadata") or {}),
            )
            trace = TraceCandidate(
                problem_id=row["problem_id"],
                response=row["response"],
                method_id=row["method_id"],
                teacher=row.get("teacher", "unknown"),
                concision_tokens=row.get("concision_tokens", 0),
            )
            if problem.partition in PROMPT_ONLY_PARTITIONS:
                ok, reasons = self.backend.verifier.verify_problem(problem)
            else:
                ok, reasons = self.backend.verifier.verify_steps(problem, trace)
            out_row = {
                **row,
                "step_verified": ok,
                "verified": ok,
                "rejection_reasons": reasons,
            }
            if not ok:
                with write_lock:
                    _append_jsonl(self.quarantine_path, out_row)
            return out_row

        if workers <= 1 or len(rows) < 8:
            out = [_verify_one(row) for row in rows]
        else:
            out = [None] * len(rows)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_verify_one, row): i for i, row in enumerate(rows)}
                for fut in as_completed(futs):
                    out[futs[fut]] = fut.result()
        _write_jsonl(self.stages_dir / "step_verify.jsonl", out)

    def stage_concision(self) -> None:
        rows = _read_jsonl(self.stages_dir / "step_verify.jsonl")
        out = []
        for row in rows:
            if not row.get("step_verified"):
                out.append(row)
                continue
            if row.get("partition") in PROMPT_ONLY_PARTITIONS:
                out.append(
                    {
                        **row,
                        "concision_tokens": 0,
                        "verified": True,
                        "step_verified": True,
                    }
                )
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
                    (extract_think(edited.response) or "").split()
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
        reference_texts: list[str] = []
        for raw_path in self.config.decontam_reference_paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"decontam reference not found: {path}")
            if path.suffix.lower() == ".jsonl":
                for ref in _read_jsonl(path):
                    text = "\n".join(
                        str(ref.get(key) or "") for key in ("prompt", "response")
                    ).strip()
                    if text:
                        reference_texts.append(text)
            else:
                reference_texts.extend(
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        # Within-batch handling avoids self-rejecting alternate traces of the
        # same problem; configured references enforce cross-run isolation.
        deco = decontaminate_records(rows, reference_texts=reference_texts)
        out = []
        for row, d in zip(rows, deco):
            aq = score_arabic_record(row)
            # Prefer scoring the think body for Arabic gates when present.
            row = {
                **row,
                "decontam": d,
                "arabic_qa": aq,
                "gate_pass": bool(row.get("verified"))
                and d.get("status") == "clean"
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
            selected_alt = None
            if alts and self.config.max_alternate_methods > 0:
                selected_alt = alts[0]
                selected.append(selected_alt)
            for c in cands:
                if c is primary or c is selected_alt:
                    continue
                if c not in good:
                    _append_jsonl(self.quarantine_path, c)

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
            metadata = dict(row.get("metadata") or {})
            is_frozen_replay = (
                self.config.backend == "external"
                and self.config.external_config.get("mode") == "replay"
            )
            if is_frozen_replay:
                metadata = {}
            else:
                metadata.update(
                    {
                        "decontam": dict(row.get("decontam") or {}),
                        "arabic_qa": dict(row.get("arabic_qa") or {}),
                    }
                )
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
                "lineage": {
                    "generator_version": "rlvr_synth",
                    "seed": row.get("seed") or (row.get("metadata") or {}).get("seed"),
                    "template_family": (row.get("metadata") or {}).get("template_family"),
                },
                "metadata": metadata,
            }
            if part in {"sft_train", "sft_eval", "rejection_sft"}:
                verifier_result = bool(
                    row.get("verified") and row.get("gate_pass")
                )
                if not verifier_result:
                    raise RuntimeError(
                        f"selected SFT row {row.get('problem_id')} is not verified"
                    )
                corpora.setdefault(part, []).append(
                    {
                        **base,
                        "response": row["response"],
                        "verifier_result": verifier_result,
                        "trace_audit": {
                            "format_ok": bool(row.get("verified")),
                            "step_verified": bool(row.get("step_verified")),
                            "concision_tokens": row.get("concision_tokens", 0),
                            "method_id": row.get("method_id"),
                            "rejection_reasons": list(
                                row.get("rejection_reasons") or []
                            ),
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
