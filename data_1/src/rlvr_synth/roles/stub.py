from __future__ import annotations
import hashlib
import json
from typing import Any
from rlvr_contracts.answer_spec import parse_answer_spec
from rlvr_contracts.response import parse_response
from rlvr_contracts.verifiers import verify_answer
from rlvr_synth.roles.protocols import ArabicEditor, IndependentVerifier, LatentProblem, ProblemGenerator, RenderedProblem, TraceCandidate, TraceTeacher

def _stable_id(*parts: Any) -> str:
    h = hashlib.sha256('|'.join((str(p) for p in parts)).encode('utf-8')).hexdigest()[:12]
    return h

class StubProblemGenerator:
    DOMAINS = ('gsm8k', 'math', 'math_comp', 'logic')

    def generate_family(self, domain: str, seed: int) -> LatentProblem:
        if domain not in self.DOMAINS:
            raise ValueError(f'unsupported stub generation domain: {domain!r}')
        a = 11 + seed * 17 % 887
        b = 13 + seed * 29 % 907
        if domain == 'logic':
            answer_spec = {'type': 'logic_json', 'canonical': {'color': 'red', 'size': (a + seed) % 5, 'id': seed % 10007}}
            latent = {'kind': 'logic_attr', 'a': a, 'b': b, 'seed': seed}
        else:
            answer_spec = {'type': 'integer', 'canonical': a + b}
            latent = {'kind': 'sum', 'a': a, 'b': b, 'seed': seed}
        family_id = f'fam_{domain}_{_stable_id(domain, seed, a, b)}'
        return LatentProblem(family_id=family_id, domain=domain, answer_spec=parse_answer_spec(answer_spec).to_dict(), latent=latent)

    def render_arabic(self, latent: LatentProblem, seed: int) -> RenderedProblem:
        if latent.latent.get('kind') == 'logic_attr':
            expected = parse_answer_spec(latent.answer_spec).ground_truth_structured
            example = json.dumps(expected, ensure_ascii=False, sort_keys=True)
            prompt = (
                f"مسألة رقم {seed}: صنف العنصر وفق البيانات الآتية: {example}. "
                "أعد كائن JSON مطابقًا بهذه الحقول والقيم."
            )
        else:
            a, b = (latent.latent['a'], latent.latent['b'])
            prompt = f'مسألة رقم {seed}: ما مجموع العددین {a} و {b}؟'
        problem_id = f'prob_{_stable_id(latent.family_id, seed, prompt)}'
        return RenderedProblem(problem_id=problem_id, family_id=latent.family_id, domain=latent.domain, partition=latent.partition, prompt=prompt, answer_spec=latent.answer_spec, provenance={'source': 'stub', 'seed': seed}, lineage={'generator_version': 'stub-v1'})

class StubTraceTeacher:

    def sample_traces(self, problem: RenderedProblem, n: int, seed: int) -> list[TraceCandidate]:
        canon = problem.answer_spec['canonical']
        if problem.answer_spec['type'] == 'logic_json':
            answer_text = json.dumps(canon, ensure_ascii=False, sort_keys=True)
            think = f'نحدد الخصائص المعطاة في المسألة.\nاللون أحمر والحجم كما هو مذكور.\nالناتج = {answer_text}'
        else:
            a = problem.metadata.get('a')
            think = f'نقرأ العددين في المسألة.\nنجمعهما خطوة بخطوة.\nإذن الناتج = {canon}'
            answer_text = str(canon)
        traces: list[TraceCandidate] = []
        for i in range(n):
            if i == 1:
                body = '\n'.join(reversed(think.splitlines()))
                method = 'alt_order'
            else:
                body = think
                method = 'direct'
            response = f'<think>\n{body}\n</think>\n<answer>{answer_text}</answer>'
            traces.append(TraceCandidate(problem_id=problem.problem_id, response=response, method_id=method, teacher='stub', concision_tokens=len(body.split()), metadata={'seed': seed, 'i': i}))
        return traces

class StubIndependentVerifier:

    def verify_problem(self, problem: RenderedProblem) -> tuple[bool, list[str]]:
        try:
            parse_answer_spec(problem.answer_spec)
        except Exception as exc:
            return (False, [f'answer_spec:{exc}'])
        if not problem.prompt.strip():
            return (False, ['empty_prompt'])
        return (True, [])

    def verify_trace(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        parsed = parse_response(trace.response)
        reasons = list(parsed.errors)
        if not parsed.format_ok:
            return (False, reasons or ['format_fail'])
        vr = verify_answer(trace.response, problem.answer_spec, from_completion=True)
        if not vr.ok:
            reasons.append(vr.reason)
            return (False, reasons)
        return (True, [])

    def verify_steps(self, problem: RenderedProblem, trace: TraceCandidate) -> tuple[bool, list[str]]:
        ok, reasons = self.verify_trace(problem, trace)
        if not ok:
            return (False, reasons)
        think = parse_response(trace.response).think or ''
        lines = [ln for ln in think.splitlines() if ln.strip()]
        if len(lines) < 2:
            return (False, ['insufficient_steps'])
        return (True, [])

class StubArabicEditor:

    def edit(self, problem: RenderedProblem, trace: TraceCandidate) -> TraceCandidate:
        return trace

class StubRoleBackend:

    def __init__(self) -> None:
        self.problem_generator: ProblemGenerator = StubProblemGenerator()
        self.trace_teacher: TraceTeacher = StubTraceTeacher()
        self.verifier: IndependentVerifier = StubIndependentVerifier()
        self.arabic_editor: ArabicEditor = StubArabicEditor()
