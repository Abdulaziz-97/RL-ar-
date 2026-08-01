#!/usr/bin/env python3
"""CPU-only static/regression gates for V4 production hardening.

GPU smoke / 2-GPU DDP / 20-step canaries are documented below and skipped
automatically when CUDA is unavailable.
"""

from __future__ import annotations

import ast
import compileall
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def test_compileall_src():
    assert compileall.compile_dir(str(SRC / "rlvr_pipeline"), quiet=1)


def test_import_rlvr_pipeline_no_vllm():
    # Package import must not require vLLM.
    import rlvr_pipeline  # noqa: F401
    assert "vllm" not in sys.modules


def test_no_session_debug_instrumentation_left():
    for path in (SRC / "rlvr_pipeline").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "a273d4" not in text, path
        assert "#region agent log" not in text, path
    hunt = REPO / "scripts" / "debug_v4_pipeline_hunt.py"
    assert not hunt.exists()


def test_preflight_module_importable():
    from rlvr_pipeline import preflight, checkpoint_integrity, lineage  # noqa: F401


@pytest.mark.skipif(not os.environ.get("RLVR_RUN_GPU_SMOKE"), reason="set RLVR_RUN_GPU_SMOKE=1")
def test_gpu_smoke_placeholder():
    import torch

    assert torch.cuda.is_available()
    # Real smoke is launched via scripts/run_v4_gpu_gates.sh on Vast.ai.


def test_shell_script_has_no_pkill_python():
    src = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")
    assert "pkill -9 -f python" not in src


def test_pass8_defaults_hf_when_instruction_lineage():
    src = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")
    assert "defaulting PASS8_BACKEND=hf" in src
    assert "SFT_HAS_INSTRUCTION" in src
    cal = (REPO / "data_1" / "scripts" / "calibrate_v4_pass8.py").read_text(encoding="utf-8")
    assert "_merge_instruction_base_to_dir" in cal
    assert "Fail-closed: vLLM pass@8" in cal


def test_lineage_loader_fail_closed_on_skipped_instruction_merge():
    src = (SRC / "rlvr_pipeline" / "lineage.py").read_text(encoding="utf-8")
    assert "_resolve_instruction_for_trainable" in src
    assert "Fail-closed lineage load" in src
    assert "merge_instruction=False" in src


def test_sft_bench_waiter_uses_generative_t06_path():
    waiter = (REPO / "outputs" / "vast_sft_bench_waiter.py").read_text(encoding="utf-8")
    assert "run_araeval_generative.py" in waiter
    assert "--instruction-adapter" in waiter or "instruction-adapter" in waiter
    assert "scripts/run_araeval.py" not in waiter
    assert "--engine" in waiter and "hf" in waiter
