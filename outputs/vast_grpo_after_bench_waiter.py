#!/usr/bin/env python3
"""After lineage-correct SFT panel completes → launch GRPO (V5 overnight).

Order enforced:
  pass@8 (external) → curate waiter → THIS waiter waits for generative SFT summary
  → torchrun GRPO with T06+SFT lineage.

Does NOT start during pass@8 or SFT bench GPU use.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/RL-ar-")
CONFIG = REPO / "configs/qwen_4b_2x5090_v4_sota.yaml"
SFT_OUT = Path("/workspace/outputs/sft_coldstart_v4_v5_20260801_174829")
RLVR_DATA = REPO / "data/arabic_reasoning_rlvr_v5.jsonl"
GRPO_OUT = Path("/workspace/outputs/qwen_4b_2x5090_v4_run_v5_20260801_174829")
SFT_PANEL = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5")
LOG = Path("/workspace/outputs/v5_grpo_after_bench.log")
STATUS = Path("/workspace/outputs/v5_grpo_after_bench_STATUS.json")
PIDFILE = Path("/workspace/outputs/v5_grpo_after_bench_waiter.pid")
GRPO_PIDFILE = Path("/workspace/outputs/v5_grpo_train.pid")
GRPO_HOLD = Path("/workspace/outputs/GRPO_HOLD_UNTIL_PASS8")
PASS8_BALANCED_STATUS = Path("/workspace/outputs/pass8_balanced2k_STATUS.json")

POLL_S = 90
MAX_WAIT_S = 8 * 3600
MEM_FREE_MIB = 2500
NUM_GPUS = 2


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()


def pgrep(pat: str) -> str:
    try:
        return sh(f"pgrep -af '{pat}' | grep -v pgrep | grep -v v5_grpo_after_bench || true")
    except Exception:
        return ""


def write_status(status: str, **extra) -> None:
    payload = {"status": status, "ts": ts(), **extra}
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[grpo-waiter] {status} {json.dumps(extra, default=str)[:300]}", flush=True)


def gpu_mem() -> list[int]:
    raw = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    return [int(x.strip()) for x in raw.splitlines() if x.strip()]


def gpus_free() -> bool:
    if pgrep("calibrate_v4_pass8").strip():
        return False
    if pgrep("run_araeval").strip():
        return False
    if pgrep("rlvr_pipeline.cli train").strip():
        return False
    mem = gpu_mem()
    return bool(mem) and all(m < MEM_FREE_MIB for m in mem)


def sft_panel_ready() -> bool:
    summary = SFT_PANEL / "summary.json"
    if not summary.is_file():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("evaluation_mode") == "generative":
        tasks = data.get("task_results") or {}
        return "araeval_aramath" in tasks and "araeval_ifeval" in tasks
    lineage = data.get("lineage_key") or {}
    if isinstance(lineage, dict) and lineage.get("instruction_adapter"):
        tasks = data.get("task_results") or {}
        return "araeval_aramath" in tasks and "araeval_ifeval" in tasks
    return False


def rlvr_ready() -> bool:
    return RLVR_DATA.is_file() and RLVR_DATA.stat().st_size > 0


def pass8_balanced_done() -> bool:
    if not PASS8_BALANCED_STATUS.is_file():
        return False
    try:
        data = json.loads(PASS8_BALANCED_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("state") == "done"


def grpo_hold_active() -> bool:
    return GRPO_HOLD.is_file() and not pass8_balanced_done()


def grpo_running() -> bool:
    return bool(pgrep("rlvr_pipeline.cli train").strip())


def launch_grpo() -> int:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    if not SFT_OUT.is_dir():
        raise FileNotFoundError(SFT_OUT)
    if not rlvr_ready():
        raise FileNotFoundError(f"missing promoted RLVR: {RLVR_DATA}")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO}/src:{REPO}:{REPO}/data_1:{REPO}/data_1/src:"
        + env.get("PYTHONPATH", "")
    )
    env["HF_HOME"] = env.get("HF_HOME", "/workspace/.hf_cache")
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "arabic-reasoning-rlvr-v4")

    py = REPO / ".venv/bin/python"
    if not py.exists():
        py = Path("/venv/main/bin/python")

    GRPO_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "-m",
        "rlvr_pipeline.cli",
        "train",
        "--config",
        str(CONFIG),
        "--output",
        str(GRPO_OUT),
        "--sft-checkpoint",
        str(SFT_OUT),
    ]
    with LOG.open("a", encoding="utf-8") as lf:
        lf.write(f"\n===GRPO_LAUNCH {ts()}===\n")
        lf.write("cmd=" + " ".join(cmd) + "\n")
        lf.write(f"rlvr={RLVR_DATA} lines=")
        lf.write(str(sum(1 for _ in RLVR_DATA.open(encoding='utf-8', errors='replace'))) + "\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    GRPO_PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    write_status("grpo_launched", grpo_pid=proc.pid, output=str(GRPO_OUT))
    return proc.pid


def main() -> int:
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    write_status("waiting_sft_panel")

    started = time.time()
    while True:
        elapsed = int(time.time() - started)
        if elapsed > MAX_WAIT_S:
            write_status("timeout", elapsed_s=elapsed)
            return 2

        if grpo_running():
            write_status("grpo_already_running", elapsed_s=elapsed)
            return 0

        if not sft_panel_ready():
            write_status(
                "waiting_sft_panel",
                elapsed_s=elapsed,
                sft_summary_exists=(SFT_PANEL / "summary.json").is_file(),
                rlvr_ready=rlvr_ready(),
            )
            time.sleep(POLL_S)
            continue

        if not rlvr_ready():
            write_status(
                "waiting_rlvr_promote",
                elapsed_s=elapsed,
                note="SFT panel done but arabic_reasoning_rlvr_v5.jsonl missing",
            )
            time.sleep(POLL_S)
            continue

        if grpo_hold_active():
            write_status(
                "waiting_pass8_balanced2k",
                elapsed_s=elapsed,
                hold_file=str(GRPO_HOLD),
                pass8_state=(
                    json.loads(PASS8_BALANCED_STATUS.read_text(encoding="utf-8")).get("state")
                    if PASS8_BALANCED_STATUS.is_file()
                    else "missing"
                ),
            )
            time.sleep(POLL_S)
            continue

        if not gpus_free():
            write_status(
                "waiting_gpus",
                elapsed_s=elapsed,
                mem_used_mib=gpu_mem(),
            )
            time.sleep(POLL_S)
            continue

        print(f"[grpo-waiter] launching GRPO {ts()}", flush=True)
        try:
            launch_grpo()
        except Exception as exc:
            write_status("grpo_launch_failed", error=str(exc), elapsed_s=elapsed)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
