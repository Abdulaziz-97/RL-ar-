from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

@dataclass
class LatentProblem:
    family_id: str
    domain: str
    answer_spec: dict[str, Any]
    latent: dict[str, Any] = field(default_factory=dict)
    partition: str = 'sft_train'

@dataclass
class RenderedProblem:
    problem_id: str
    family_id: str
    domain: str
    partition: str
    prompt: str
    answer_spec: dict[str, Any]
    verifier_type: str = 'python'
    verifier_version: str = 'rlvr-contracts-verifiers-v1'
    provenance: dict[str, Any] = field(default_factory=dict)
    licensing: dict[str, Any] = field(default_factory=lambda: {'license': 'internal'})
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceCandidate:
    problem_id: str
    response: str
    method_id: str
    teacher: str
    verified: bool = False
    step_verified: bool = False
    concision_tokens: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class ProblemGenerator(Protocol):

    def generate_family(self, domain: str, seed: int) -> LatentProblem:
        ...

    def render_arabic(self, latent: LatentProblem, seed: int) -> RenderedProblem:
        ...

@runtime_checkable
class TraceTeacher(Protocol):

    def sample_traces(self, problem: RenderedProblem, n: int, seed: int) -> list[TraceCandidate]:
        ...

@runtime_checkable
class IndependentVerifier(Protocol):

    def verify_problem(self, problem: RenderedProblem) -> tuple[bool, list[str]]:
        ...

    def verify_trace(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        ...

    def verify_steps(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        ...

@runtime_checkable
class ArabicEditor(Protocol):

    def edit(self, problem: RenderedProblem, trace: TraceCandidate) -> TraceCandidate:
        ...
