#!/usr/bin/env python3
"""Wait for GPUs free, then launch 2-shard HF pass@8 on balanced 2k candidates."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

STATUS = Path("/workspace/outputs/pass8_balanced2k_STATUS.json")
LOG = Path("/workspace/outputs/pass8_balanced2k_waiter.log")
CAND = Path("/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl")
SFT = Path("/workspace/outputs/sft_coldstart_v4_v5_20260801_174829")
OUT = Path("/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8_balanced2k")
PY = "/workspace/RL-ar-/.venv/bin/python"
if not Path(PY).exists():
    PY = "/venv/main/bin/python"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kw):
    STATUS.write_text(json.dumps({"ts": time.time(), **kw}, indent=2), encoding="utf-8")


def pgrep(pat: str) -> str:
    r = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True)
    return "\n".join(
        ln for ln in (r.stdout or "").splitlines() if "pgrep" not in ln and ln.strip()
    )


def gpu_busy() -> bool:
    for pat in (
        "run_araeval_generative",
        "calibrate_v4_pass8",
        "trl.*grpo",
        "train_grpo",
        "accelerate launch",
    ):
        if pgrep(pat).strip():
            return True
    return False


def launch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("cand_shard*.jsonl"):
        p.unlink(missing_ok=True)
    for p in OUT.glob("cal_shard*.json"):
        p.unlink(missing_ok=True)

    n = sum(1 for _ in CAND.open(encoding="utf-8") if _.strip())
    log(f"launch pass@8 on {CAND} n={n} sft={SFT}")
    write_status(state="launching", n=n, out=str(OUT))

    env_base = os.environ.copy()
    env_base["HF_HOME"] = env_base.get("HF_HOME", "/workspace/hf")
    env_base["TRANSFORMERS_CACHE"] = env_base.get("TRANSFORMERS_CACHE", "/workspace/hf")
    env_base["TOKENIZERS_PARALLELISM"] = "false"
    env_base["PYTHONUNBUFFERED"] = "1"

    batch = "4"
    for shard in (0, 1):
        log_path = Path(f"/workspace/outputs/pass8_balanced2k_shard{shard}.log")
        cmd = [
            PY,
            "/workspace/RL-ar-/data_1/scripts/calibrate_v4_pass8.py",
            "--candidates",
            str(CAND),
            "--sft-checkpoint",
            str(SFT),
            "--out-calibration",
            str(OUT / f"cal_shard{shard}.json"),
            "--out-candidates",
            str(OUT / f"cand_shard{shard}.jsonl"),
            "--shard-id",
            str(shard),
            "--num-shards",
            "2",
            "--backend",
            "hf",
            "--n",
            "8",
            "--temperature",
            "0.7",
            "--max-new-tokens",
            "768",
            "--prompt-batch-size",
            batch,
            "--merge-sft-lora",
        ]
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        with log_path.open("w") as lf:
            subprocess.Popen(
                cmd,
                cwd="/workspace/RL-ar-",
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log(f"launched shard={shard} log={log_path}")
    write_status(state="running", n=n, out=str(OUT), batch=batch)
    log("pass@8 launched both shards")


def main() -> int:
    log("waiter start")
    write_status(state="waiting_gpus")
    while gpu_busy():
        busy = []
        for pat in ("run_araeval_generative", "calibrate_v4_pass8", "trl", "accelerate"):
            if pgrep(pat).strip():
                busy.append(pat)
        log(f"gpus busy ({','.join(busy) or 'unknown'}); sleep 60")
        write_status(state="waiting_gpus", busy=busy)
        time.sleep(60)

    if not CAND.is_file():
        write_status(state="error", error="missing candidates")
        raise SystemExit("missing candidates")
    if not (SFT / "adapter_config.json").is_file():
        write_status(state="error", error="missing sft")
        raise SystemExit("missing sft")

    launch()
    while True:
        procs = pgrep("calibrate_v4_pass8")
        s0 = OUT / "cand_shard0.jsonl"
        s1 = OUT / "cand_shard1.jsonl"
        n0 = sum(1 for _ in s0.open(encoding="utf-8") if _.strip()) if s0.is_file() else 0
        n1 = sum(1 for _ in s1.open(encoding="utf-8") if _.strip()) if s1.is_file() else 0
        write_status(
            state="running" if procs.strip() else "done",
            shard0=n0,
            shard1=n1,
            procs=bool(procs.strip()),
        )
        if not procs.strip() and n0 > 0 and n1 > 0:
            log(f"DONE shards {n0}+{n1}")
            write_status(state="done", shard0=n0, shard1=n1)
            return 0
        if not procs.strip() and (n0 == 0 or n1 == 0):
            log(f"pass8 exited early shard0={n0} shard1={n1}")
            write_status(state="error", shard0=n0, shard1=n1, error="early_exit")
            return 1
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
