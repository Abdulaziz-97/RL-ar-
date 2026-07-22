from __future__ import annotations
import argparse
import json
from pathlib import Path
from rlvr_contracts.verifiers import VERIFIER_REGISTRY_VERSION
from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator
from rlvr_synth.release.manifest import build_manifest, write_manifest
from rlvr_synth.release.ship_gate import run_ship_gate, write_ship_report

PACK_ROOT = Path(__file__).resolve().parents[3]


def cmd_run(args: argparse.Namespace) -> int:
    cfg = SynthConfig.from_yaml(args.config) if args.config else SynthConfig()
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.mode:
        cfg.mode = args.mode
    if args.n_families:
        cfg.n_families = args.n_families
    if args.backend:
        cfg.backend = args.backend
    if args.external_module:
        cfg.external_module = args.external_module
        cfg.backend = 'external'
    if cfg.external_module and not Path(cfg.external_module).is_absolute():
        cfg.external_module = str((PACK_ROOT / cfg.external_module).resolve())
    orch = SynthOrchestrator(cfg)
    result = orch.run(resume=not args.no_resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

def cmd_ship(args: argparse.Namespace) -> int:
    work = Path(args.work_dir)
    corpora_dir = work / 'release_corpora'
    corpora = {}
    schema_kinds = {'sft_train': 'sft_trace', 'sft_eval': 'sft_trace', 'rejection_sft': 'sft_trace', 'rlvr_train': 'rlvr_prompt', 'rlvr_eval': 'rlvr_prompt', 'private_eval': 'evaluation'}
    for name, kind in schema_kinds.items():
        path = corpora_dir / f'{name}.jsonl'
        if path.exists() and path.stat().st_size > 0:
            corpora[name] = path
    artifacts = []
    for name, path in corpora.items():
        artifacts.append((name, f'release_corpora/{name}.jsonl', path))
    release_id = args.release_id or f'stub_{work.name}'
    manifest = build_manifest(release_id, corpus_version=args.corpus_version or release_id, verifier_registry_version=VERIFIER_REGISTRY_VERSION, artifacts=artifacts, notes={'backend': 'stub_or_external', 'credential_free': True})
    # Immutable releases need distinct directories; otherwise a second
    # release_id can never be written despite the error suggesting it.
    release_dir = work / 'release' / release_id
    man_path = write_manifest(manifest, release_dir)
    report = run_ship_gate(corpora=corpora, manifest=manifest, root=work, schema_kind_by_corpus=schema_kinds)
    write_ship_report(report, release_dir / 'ship_gate_report.json')
    print(json.dumps({'manifest': str(man_path), 'passed': report.passed, 'errors': report.errors}, ensure_ascii=False, indent=2))
    return 0 if report.passed else 1

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(prog='rlvr-synth')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_run = sub.add_parser('run', help='Run resumable synth DAG')
    p_run.add_argument('--config', type=str, default='')
    p_run.add_argument('--work-dir', type=str, default='')
    p_run.add_argument('--mode', type=str, default='')
    p_run.add_argument('--n-families', type=int, default=0)
    p_run.add_argument('--backend', type=str, default='')
    p_run.add_argument('--external-module', type=str, default='')
    p_run.add_argument('--no-resume', action='store_true')
    p_run.set_defaults(func=cmd_run)
    p_ship = sub.add_parser('ship', help='Build manifest + run ship gate')
    p_ship.add_argument('--work-dir', type=str, required=True)
    p_ship.add_argument('--release-id', type=str, default='')
    p_ship.add_argument('--corpus-version', type=str, default='')
    p_ship.set_defaults(func=cmd_ship)
    args = parser.parse_args(argv)
    return int(args.func(args))
if __name__ == '__main__':
    raise SystemExit(main())
