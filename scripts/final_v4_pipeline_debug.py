#!/usr/bin/env python3
"""Final debug gate: data / training / reward scripts before Vast.ai launch.

Does NOT require a shippable 4k+4k corpus (regen path produces that).
Exit 0 only if script imports, reward contracts, and unit suites pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "data_1"


def _ok(msg: str) -> None:
    print(f"  OK  {msg}", flush=True)


def _fail(failures: list[str], msg: str) -> None:
    print(f" FAIL {msg}", flush=True)
    failures.append(msg)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_data_scripts(failures: list[str]) -> None:
    print("\n=== DATA SCRIPTS ===", flush=True)
    scripts = [
        "run_pipeline.py",
        "select_sft_v4_release.py",
        "calibrate_v4_pass8.py",
        "curate_v4_rlvr.py",
        "promote_v4_release.py",
        "export_human_review_packs.py",
        "verify_master_v4_datasets_complete.py",
        "run_v4_master_orchestrator.py",
    ]
    for name in scripts:
        path = PACK / "scripts" / name
        if not path.exists():
            _fail(failures, f"missing {path}")
            continue
        try:
            src = path.read_text(encoding="utf-8")
            compile(src, str(path), "exec")
            _ok(f"compile {name}")
        except Exception as exc:  # noqa: BLE001
            _fail(failures, f"compile {name}: {exc}")

    # Parallel multi_trace smoke with stub backend (no API).
    sys.path[:0] = [str(PACK / "src"), str(PACK / "vendor"), str(PACK)]
    from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator
    from rlvr_synth.roles.protocols import TraceCandidate

    with tempfile.TemporaryDirectory() as td:
        cfg = SynthConfig(
            mode="cold_start",
            n_families=8,
            domains=["gsm8k", "math", "math_comp", "logic"],
            traces_per_problem=1,
            seed=7,
            work_dir=td,
            backend="stub",
            partitions={"sft_train": 1.0},
            external_config={"teacher_workers": 4},
        )
        orch = SynthOrchestrator(cfg)
        # Force parallel path even on stub by monkeypatching sample_traces latency-free.
        calls = {"n": 0}

        def _sample(problem, n, seed):  # noqa: ANN001
            calls["n"] += 1
            return [
                TraceCandidate(
                    problem_id=problem.problem_id,
                    response="<think>1+1=2</think>\n<answer>2</answer>",
                    method_id="stub_0",
                    teacher="stub",
                    concision_tokens=3,
                )
                for _ in range(max(1, n))
            ]

        orch.backend.trace_teacher.sample_traces = _sample  # type: ignore[method-assign]
        orch.stage_family_partition()
        orch.stage_latent_spec()
        orch.stage_deterministic_solve()
        orch.stage_arabic_render()
        t0 = time.perf_counter()
        orch.stage_multi_trace()
        elapsed = time.perf_counter() - t0
        mt = (Path(td) / "stages" / "multi_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if len(mt) != 8:
            _fail(failures, f"multi_trace rows={len(mt)} expected 8")
        else:
            _ok(f"parallel multi_trace stub n=8 workers=4 in {elapsed:.3f}s calls={calls['n']}")

    vast = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")
    for needle in (
        "TEACHER_WORKERS",
        "TEACHER_MODEL",
        "--workers",
        "DATAGEN_CANARY",
        "select_sft_v4_release.py",
        "calibrate_v4_pass8.py",
        "curate_v4_rlvr.py",
        "promote_v4_release.py",
    ):
        if needle not in vast:
            _fail(failures, f"setup_and_run_vastai.sh missing {needle}")
        else:
            _ok(f"vastai wires {needle}")


def check_training(failures: list[str]) -> None:
    print("\n=== TRAINING SCRIPTS ===", flush=True)
    sys.path.insert(0, str(REPO / "src"))
    try:
        from rlvr_pipeline.config import RLVRConfig
        from rlvr_pipeline.rewards import diagnose_reward_weights

        cfg = RLVRConfig.from_yaml(REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml")
        if cfg.sft_num_train_epochs is None or cfg.sft_num_train_epochs < 1:
            _fail(failures, "sft_num_train_epochs missing")
        else:
            _ok(f"sft_epochs={cfg.sft_num_train_epochs} grpo_max_steps={cfg.max_steps}")
        if set(cfg.lora_target_modules) & {"embed_tokens", "lm_head"}:
            _fail(failures, "forbidden embed/lm_head LoRA targets")
        else:
            _ok(f"lora_targets={len(cfg.lora_target_modules)} linear-only")
        if cfg.beta <= 0:
            _fail(failures, f"beta={cfg.beta} expected >0 for SFT-anchored GRPO")
        else:
            _ok(f"beta={cfg.beta}")
        if cfg.curriculum_schedule_type != "gaussian":
            _fail(
                failures,
                f"curriculum={cfg.curriculum_schedule_type} expected gaussian "
                "(pass@8 bins exist before GRPO)",
            )
        else:
            _ok("curriculum=gaussian")
        report = diagnose_reward_weights(cfg.reward_weights)
        _ok(f"reward_weights ok keys={list(report) if isinstance(report, dict) else 'ok'}")
    except Exception as exc:  # noqa: BLE001
        _fail(failures, f"training config/reward: {exc}")

    for mod_name in ("rlvr_pipeline.cli", "rlvr_pipeline.preflight", "rlvr_pipeline.checkpoint_integrity"):
        try:
            __import__(mod_name)
            _ok(f"import {mod_name}")
        except Exception as exc:  # noqa: BLE001
            _fail(failures, f"import {mod_name}: {exc}")


def check_rewards(failures: list[str]) -> None:
    print("\n=== REWARD SCRIPTS ===", flush=True)
    try:
        from rlvr.reward_composer import compose_reward
        from rlvr_pipeline.rewards import ALL_REWARD_FUNCS

        score = compose_reward("<think>1+1=2</think>\n<answer>2</answer>", 2, "math")
        _ok(f"compose_reward sample score={score!r}")
        if not ALL_REWARD_FUNCS:
            _fail(failures, "ALL_REWARD_FUNCS empty")
        else:
            _ok(f"ALL_REWARD_FUNCS n={len(ALL_REWARD_FUNCS)}")
    except Exception as exc:  # noqa: BLE001
        _fail(failures, f"reward smoke: {exc}")


def run_unit_suites(failures: list[str]) -> None:
    print("\n=== UNIT SUITES ===", flush=True)
    cmds = [
        [sys.executable, "-m", "pytest", str(PACK / "tests" / "test_v4_datagen_quality.py"), "-q", "--tb=line"],
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO / "tests" / "test_v4_reward_contract.py"),
            str(REPO / "tests" / "test_v4_production_recipe.py"),
            "-q",
            "--tb=line",
        ],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(PACK / "src"), str(PACK / "vendor"), str(PACK), env.get("PYTHONPATH", "")]
    )
    for cmd in cmds:
        print("  RUN", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
        if res.returncode != 0:
            _fail(failures, f"pytest failed: {cmd[-2] if len(cmd) > 2 else cmd}\n{res.stdout[-800:]}\n{res.stderr[-800:]}")
        else:
            tail = (res.stdout or "").strip().splitlines()[-1:] or ["passed"]
            _ok(f"pytest {' '.join(Path(c).name for c in cmd if c.endswith('.py'))} :: {tail[-1]}")


def main() -> int:
    print("=" * 72, flush=True)
    print(" FINAL V4 PIPELINE DEBUG (data + training + reward)", flush=True)
    print("=" * 72, flush=True)
    failures: list[str] = []
    check_data_scripts(failures)
    check_training(failures)
    check_rewards(failures)
    run_unit_suites(failures)

    print("\n" + "=" * 72, flush=True)
    if failures:
        print(f"FAILED ({len(failures)})", flush=True)
        for f in failures:
            print(f" - {f}", flush=True)
        return 1
    print("ALL DEBUG GATES PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
