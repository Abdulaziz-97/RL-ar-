#!/usr/bin/env python3
"""Durable on-Vast waiter: free GPUs after pass@8 → lineage-correct SFT panel.

Uses official_eval/run_araeval_generative.py with T06 instruction merge + SFT
adapter on the HF engine (same pattern as e2e stage 8). Never launches bare
run_araeval.py against raw Qwen + SFT LoRA.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ADAPTER = Path("/workspace/outputs/sft_coldstart_v4_v5_20260801_174829")
OUT = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5")
LOG = Path("/workspace/outputs/sft_v5_aramath_araifeval_panel.log")
STATUS = Path("/workspace/outputs/sft_v5_aramath_araifeval_STATUS.json")
PIDFILE = Path("/workspace/outputs/sft_v5_aramath_araifeval_waiter.pid")
BENCH_PIDFILE = Path("/workspace/outputs/sft_v5_aramath_araifeval.pid")
GRPO_SUMMARY = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_grpo_v3/summary.json")
EVAL_SCRIPT = Path("/workspace/RL-ar-/official_eval/run_araeval_generative.py")
DEFAULT_INSTRUCTION = "aziz9788/T06__qwen35-mixed-v6-lr1e5"
BASE_MODEL = "Qwen/Qwen3.5-4B"

POLL_S = 60
MAX_WAIT_S = 6 * 3600
MEM_FREE_MIB = 2500


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()


def pgrep(pat: str) -> str:
    try:
        return sh(f"pgrep -af '{pat}' | grep -v pgrep | grep -v vast_sft_bench_waiter || true")
    except Exception:
        return ""


def resolve_instruction() -> str:
    lineage = ADAPTER / "lineage.json"
    if lineage.is_file():
        try:
            data = json.loads(lineage.read_text(encoding="utf-8"))
            instr = (data.get("instruction_adapter") or "").strip()
            if instr:
                return instr
        except Exception:
            pass
    return DEFAULT_INSTRUCTION


def write_status(status: str, **extra) -> None:
    payload = {
        "status": status,
        "ts": ts(),
        "adapter": str(ADAPTER),
        "instruction_adapter": resolve_instruction(),
        "log": str(LOG),
        "out": str(OUT),
        "waiter_pid": os.getpid(),
        "eval_script": str(EVAL_SCRIPT),
    }
    payload.update(extra)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[status] {status} "
        f"{json.dumps({k: payload.get(k) for k in ('pass8', 'gpus_free', 'bench_pid', 'elapsed_s') if k in payload})}",
        flush=True,
    )


def gpu_mem_used() -> list[int]:
    raw = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    return [int(x.strip()) for x in raw.splitlines() if x.strip()]


def gpus_free() -> bool:
    if pgrep("calibrate_v4_pass8").strip():
        return False
    mem = gpu_mem_used()
    return bool(mem) and all(m < MEM_FREE_MIB for m in mem)


def correct_bench_running() -> bool:
    return bool(pgrep("run_araeval_generative.py").strip())


def wrong_lineage_bench_running() -> bool:
    """Bare run_araeval.py without T06 merge — must not count as the SFT panel."""
    return bool(pgrep("run_araeval.py").strip()) and not correct_bench_running()


def bench_running() -> bool:
    return correct_bench_running()


def summary_ready() -> bool:
    """Accept only lineage-correct generative panel summaries (T06 + both tasks)."""
    path = OUT / "summary.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Reject log-likelihood / bare-adapter panels (no instruction merge).
    lineage_key = data.get("lineage_key") or {}
    if isinstance(lineage_key, dict) and lineage_key.get("instruction_adapter"):
        tasks = data.get("task_results") or {}
        return "araeval_aramath" in tasks and "araeval_ifeval" in tasks
    if data.get("evaluation_mode") == "generative":
        tasks = data.get("task_results") or {}
        return "araeval_aramath" in tasks and "araeval_ifeval" in tasks
    return False


def invalidate_stale_summary() -> None:
    """Move aside incomplete / wrong-lineage summaries so the correct bench can run."""
    path = OUT / "summary.json"
    if not path.is_file():
        return
    if summary_ready():
        return
    bak = OUT / f"summary.stale_wrong_lineage_{int(time.time())}.json"
    path.rename(bak)
    ckpt = OUT / "checkpoint.json"
    if ckpt.is_file():
        ckpt.rename(OUT / f"checkpoint.stale_{int(time.time())}.json")
    print(f"[vast-waiter] invalidated stale summary -> {bak}", flush=True)


def _sft_raw_from_summary(sft: dict) -> dict:
    """Normalize generative or log-likelihood summary into a flat score map."""
    raw = dict(sft.get("raw_percent") or {})
    if raw:
        return raw
    task_results = sft.get("task_results") or {}
    out: dict = {}
    aramath = task_results.get("araeval_aramath") or {}
    if "accuracy" in aramath:
        out["aramath"] = aramath["accuracy"]
    ifeval = task_results.get("araeval_ifeval") or {}
    if "accuracy" in ifeval:
        # Generative IFEval reports a single strict-style accuracy.
        out["ifeval_prompt_strict"] = ifeval["accuracy"]
        out["ifeval_instruction_strict"] = ifeval["accuracy"]
    if sft.get("overall_generative_accuracy") is not None:
        out["overall_generative_accuracy"] = sft["overall_generative_accuracy"]
    return out


def report_done() -> dict:
    sft = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    grpo = json.loads(GRPO_SUMMARY.read_text(encoding="utf-8")) if GRPO_SUMMARY.is_file() else {}
    raw = _sft_raw_from_summary(sft)
    grow = grpo.get("raw_percent", {})
    instruction = resolve_instruction()
    report = {
        "status": "done",
        "ts": ts(),
        "sft_raw": raw,
        "grpo_v3_raw": grow,
        "compare": {
            "aramath": {
                "sft": raw.get("aramath"),
                "grpo_v3": grow.get("aramath"),
                "delta": (
                    None
                    if raw.get("aramath") is None or grow.get("aramath") is None
                    else raw["aramath"] - grow["aramath"]
                ),
            },
            "ifeval_prompt_strict": {
                "sft": raw.get("ifeval_prompt_strict"),
                "grpo_v3": grow.get("ifeval_prompt_strict"),
            },
            "ifeval_instruction_strict": {
                "sft": raw.get("ifeval_instruction_strict"),
                "grpo_v3": grow.get("ifeval_instruction_strict"),
            },
        },
        "log": str(LOG),
        "out": str(OUT),
        "command": (
            f"python {EVAL_SCRIPT} --model {BASE_MODEL} "
            f"--instruction-adapter {instruction} --adapter-path {ADAPTER} "
            "--tasks araeval_aramath araeval_ifeval --engine hf "
            f"--max-lora-rank 128 --batch-size 8 --output-dir {OUT}"
        ),
    }
    STATUS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def launch_bench() -> int:
    if not EVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"missing eval script: {EVAL_SCRIPT}")
    if not (ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing adapter_config: {ADAPTER}")
    if not list(ADAPTER.glob("adapter_model.*")):
        raise FileNotFoundError(f"missing adapter weights: {ADAPTER}")

    instruction = resolve_instruction()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Fresh SFT panel (never resume GRPO_v3 checkpoint)
    for p in (OUT / "checkpoint.json", OUT / "summary.json"):
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env["HF_HOME"] = "/workspace/.hf_home"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = (
        "/workspace/RL-ar-/src:/workspace/RL-ar-:"
        + env.get("PYTHONPATH", "")
    )
    # Prefer project venv (has peft/transformers stack used by lineage loader)
    py = "/workspace/RL-ar-/.venv/bin/python"
    if not Path(py).exists():
        py = "/venv/main/bin/python" if Path("/venv/main/bin/python").exists() else "python3"

    cmd = [
        py,
        str(EVAL_SCRIPT),
        "--model",
        BASE_MODEL,
        "--instruction-adapter",
        instruction,
        "--adapter-path",
        str(ADAPTER),
        "--tasks",
        "araeval_aramath",
        "araeval_ifeval",
        "--engine",
        "hf",
        "--max-lora-rank",
        "128",
        "--batch-size",
        "8",
        "--output-dir",
        str(OUT),
    ]
    with LOG.open("a", encoding="utf-8") as lf:
        lf.write(f"\n===SFT_V5_BENCH_START {ts()}===\n")
        lf.write(f"adapter={ADAPTER}\n")
        lf.write(f"instruction_adapter={instruction}\n")
        lf.write("tasks=araeval_aramath araeval_ifeval engine=hf\n")
        lf.write("lineage=T06-merge + SFT LoRA (run_araeval_generative)\n")
        lf.write("cmd=" + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd="/workspace/RL-ar-/official_eval",
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    BENCH_PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(5)
    if proc.poll() is not None:
        tail = LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
        write_status("bench_launch_failed", exit_code=proc.returncode, log_tail=tail)
        raise RuntimeError(f"bench exited early code={proc.returncode}")
    write_status(
        "launched",
        bench_pid=proc.pid,
        command=" ".join(cmd),
        instruction_adapter=instruction,
    )
    print(f"[vast-waiter] BENCH_PID={proc.pid}", flush=True)
    return proc.pid


def main() -> int:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    print(
        f"[vast-waiter] start pid={os.getpid()} {ts()} "
        f"instruction={resolve_instruction()}",
        flush=True,
    )
    started = time.time()
    invalidate_stale_summary()

    if summary_ready():
        report_done()
        return 0

    if bench_running():
        write_status("launched_existing", note="lineage-correct generative bench running")
    elif wrong_lineage_bench_running():
        write_status(
            "waiting",
            note="legacy/wrong-lineage run_araeval.py detected; waiting for GPUs then launching generative+T06",
        )
    else:
        write_status("waiting", note="polling for pass8 GPU free")

    while True:
        elapsed = int(time.time() - started)
        if elapsed > MAX_WAIT_S:
            write_status("timeout", elapsed_s=elapsed)
            print("[vast-waiter] TIMEOUT", flush=True)
            return 2

        if summary_ready():
            print("[vast-waiter] DONE", flush=True)
            report_done()
            return 0

        pass8 = bool(pgrep("calibrate_v4_pass8").strip())
        mem = gpu_mem_used()
        free = gpus_free()
        running = bench_running()
        print(
            f"[probe] elapsed={elapsed}s pass8={int(pass8)} free={free} "
            f"bench={int(running)} mem={mem}",
            flush=True,
        )

        if running:
            write_status("running", elapsed_s=elapsed, pass8=pass8, mem_used_mib=mem, gpus_free=free)
        elif free:
            print(f"[vast-waiter] GPUs free — launching SFT bench NOW {ts()}", flush=True)
            try:
                launch_bench()
            except Exception as exc:
                print(f"[vast-waiter] launch error: {exc}", flush=True)
                write_status("bench_launch_failed", error=str(exc), elapsed_s=elapsed)
        else:
            write_status(
                "waiting",
                elapsed_s=elapsed,
                pass8=pass8,
                mem_used_mib=mem,
                gpus_free=free,
            )

        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
