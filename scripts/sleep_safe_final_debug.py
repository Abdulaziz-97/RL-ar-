#!/usr/bin/env python3
"""Sleep-safe final debug before overnight Vast.ai full pipeline.

Writes NDJSON evidence to workspace debug-a273d4.log (session a273d4).
Hypotheses covered:
  H-A: REGENERATE_DATA path missing from setup / wrong order
  H-B: promote dies overnight without SKIP_HUMAN_REVIEW
  H-C: empty SFT_RESUME_FLAG + set -u can break bash
  H-D: critical scripts/configs missing or uncompilable
  H-E: training/reward contracts broken for overnight GRPO
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO.parent / "debug-a273d4.log"
ALT = REPO / "debug-a273d4.log"
SESSION = "a273d4"
RUN_ID = os.environ.get("DEBUG_RUN_ID", "pre-sleep")


def _log(hid: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": SESSION,
        "runId": RUN_ID,
        "hypothesisId": hid,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for path in (LOG, ALT):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def main() -> int:
    failures: list[str] = []
    vast = (REPO / "scripts" / "setup_and_run_vastai.sh").read_text(encoding="utf-8")

    # #region agent log
    # H-A: regen stages present and ordered
    stages = [
        "REGENERATE_DATA",
        "run_pipeline.py",
        "--track sft",
        "select_sft_v4_release.py",
        "cli sft",
        "--track rlvr",
        "calibrate_v4_pass8.py",
        "curate_v4_rlvr.py",
        "promote_v4_release.py",
        "cli train",
    ]
    positions = {s: vast.find(s) for s in stages}
    ordered = all(
        positions[stages[i]] >= 0 and positions[stages[i]] < positions[stages[i + 1]]
        for i in range(len(stages) - 1)
        if positions[stages[i]] >= 0 and positions[stages[i + 1]] >= 0
    )
    missing = [s for s, p in positions.items() if p < 0]
    _log(
        "A",
        "sleep_debug:H-A",
        "regen stage wiring",
        {"ordered": ordered, "missing": missing, "positions": positions},
    )
    if missing or not ordered:
        failures.append(f"H-A stages missing/out-of-order: {missing} ordered={ordered}")
    # #endregion

    # #region agent log
    # H-B: overnight promote requires SKIP_HUMAN_REVIEW or report
    needs_review_gate = "SKIP_HUMAN_REVIEW" in vast and "HUMAN_REVIEW_REPORT" in vast
    hard_exit_without = "exit 2" in vast and "SKIP_HUMAN_REVIEW" in vast
    auto_skip = "auto-set SKIP_HUMAN_REVIEW=1" in vast
    _log(
        "B",
        "sleep_debug:H-B",
        "human review overnight trap",
        {
            "needs_review_gate": needs_review_gate,
            "hard_exit_without": hard_exit_without,
            "auto_skip_for_unattended": auto_skip,
            "must_export_skip": not auto_skip,
        },
    )
    if not needs_review_gate:
        failures.append("H-B promote human-review gate missing")
    if not auto_skip:
        failures.append("H-B missing auto SKIP_HUMAN_REVIEW for unattended regen")
    # #endregion

    # #region agent log
    # H-C: empty array expansion under set -u
    has_set_u = "set -euo pipefail" in vast
    resume_block = "SFT_RESUME_FLAG=()" in vast and '"${SFT_RESUME_FLAG[@]}"' in vast
    resume_safe = "DATAGEN_FRESH" in vast and resume_block is False
    _log(
        "C",
        "sleep_debug:H-C",
        "empty resume flag under set -u",
        {
            "has_set_u": has_set_u,
            "resume_block": resume_block,
            "resume_safe_if_else": resume_safe,
        },
    )
    if resume_block:
        failures.append("H-C empty SFT_RESUME_FLAG[@] still present under set -u")
    # #endregion

    # #region agent log
    # H-D: scripts compile + configs exist
    required = [
        REPO / "data_1" / "scripts" / "run_pipeline.py",
        REPO / "data_1" / "scripts" / "select_sft_v4_release.py",
        REPO / "data_1" / "scripts" / "calibrate_v4_pass8.py",
        REPO / "data_1" / "scripts" / "curate_v4_rlvr.py",
        REPO / "data_1" / "scripts" / "promote_v4_release.py",
        REPO / "data_1" / "configs" / "full_sft_6500.yaml",
        REPO / "data_1" / "configs" / "full_rlvr_8000.yaml",
        REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml",
        REPO / "scripts" / "final_v4_pipeline_debug.py",
    ]
    missing_paths = [str(p) for p in required if not p.exists()]
    compile_fails = []
    for p in required:
        if p.suffix == ".py" and p.exists():
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except Exception as exc:  # noqa: BLE001
                compile_fails.append(f"{p.name}:{exc}")
    _log(
        "D",
        "sleep_debug:H-D",
        "scripts/configs presence",
        {"missing_paths": missing_paths, "compile_fails": compile_fails},
    )
    if missing_paths or compile_fails:
        failures.append(f"H-D missing={missing_paths} compile={compile_fails}")
    # #endregion

    # #region agent log
    # H-E: training/reward contracts
    sys.path.insert(0, str(REPO / "src"))
    from rlvr_pipeline.config import RLVRConfig
    from rlvr_pipeline.rewards import diagnose_reward_weights

    cfg = RLVRConfig.from_yaml(REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml")
    reward_ok = True
    reward_err = None
    try:
        diagnose_reward_weights(cfg.reward_weights)
    except Exception as exc:  # noqa: BLE001
        reward_ok = False
        reward_err = str(exc)
    e_data = {
        "sft_epochs": cfg.sft_num_train_epochs,
        "grpo_max_steps": cfg.max_steps,
        "beta": cfg.beta,
        "curriculum": cfg.curriculum_schedule_type,
        "forbidden_lora": bool({"embed_tokens", "lm_head"} & set(cfg.lora_target_modules)),
        "reward_ok": reward_ok,
        "reward_err": reward_err,
        "teacher_workers_default_96": "TEACHER_WORKERS:-96" in vast or 'TEACHER_WORKERS:-96' in vast.replace('"', ""),
    }
    # tolerant check for ${TEACHER_WORKERS:-96}
    e_data["teacher_workers_default_96"] = bool(re.search(r"TEACHER_WORKERS:-\}?96", vast)) or "TEACHER_WORKERS:-96" in vast
    _log("E", "sleep_debug:H-E", "train/reward contracts", e_data)
    if cfg.beta <= 0 or e_data["forbidden_lora"] or not reward_ok:
        failures.append(f"H-E train/reward broken: {e_data}")
    # #endregion

    # #region agent log
    # H-F: speed knobs present (non-skipping)
    speed = {
        "TEACHER_WORKERS": "TEACHER_WORKERS" in vast,
        "VERIFY_WORKERS": "VERIFY_WORKERS" in vast,
        "DATAGEN_RESUME_MULTI_TRACE": "DATAGEN_RESUME_MULTI_TRACE" in vast,
        "DATAGEN_CANARY": "DATAGEN_CANARY" in vast,
        "num_return_sequences": "num_return_sequences"
        in (REPO / "data_1" / "scripts" / "calibrate_v4_pass8.py").read_text(encoding="utf-8"),
        "multi_trace_partial": "multi_trace.partial.jsonl"
        in (REPO / "data_1" / "src" / "rlvr_synth" / "orchestrator.py").read_text(encoding="utf-8"),
    }
    _log("F", "sleep_debug:H-F", "speed knobs without gate skip", speed)
    if not all(speed.values()):
        failures.append(f"H-F missing speed knobs: {speed}")
    # #endregion

    # Run unit suites quickly
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "data_1" / "src"), str(REPO / "data_1" / "vendor"), str(REPO / "data_1")]
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(REPO / "data_1" / "tests" / "test_v4_datagen_quality.py"),
        str(REPO / "tests" / "test_v4_reward_contract.py"),
        "-q",
        "--tb=line",
    ]
    res = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    _log(
        "G",
        "sleep_debug:pytest",
        "unit suites",
        {"returncode": res.returncode, "tail": (res.stdout or "")[-400:]},
    )
    if res.returncode != 0:
        failures.append(f"pytest failed: {(res.stderr or res.stdout)[-500:]}")

    print("=" * 72)
    if failures:
        print(f"SLEEP DEBUG FAIL ({len(failures)})")
        for f in failures:
            print(" -", f)
        _log("Z", "sleep_debug:summary", "FAIL", {"failures": failures})
        return 1
    print("SLEEP DEBUG PASS — safe to launch overnight with SKIP_HUMAN_REVIEW=1")
    _log("Z", "sleep_debug:summary", "PASS", {"ok": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
