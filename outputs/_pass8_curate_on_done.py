#!/usr/bin/env python3
"""Monitor-only: when both pass8 shards finish, curate (+ optional promote). Never relaunch."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/RL-ar-")
DATAGEN = REPO / "data_1/outputs/v4_regen/v5_20260801_174829"
OUT = Path(os.environ.get("PASS8_OUT", str(DATAGEN / "pass8_balanced2k")))
SHARDS = [OUT / f"cand_shard{i}.jsonl" for i in (0, 1)]
CURATE = REPO / "data_1/scripts/curate_v4_rlvr.py"
SELECTED = Path(os.environ.get("CURATE_OUT", str(DATAGEN / "rlvr_selected_2000.jsonl")))
LOG = Path("/workspace/outputs/pass8_curate_on_done.log")
STATUS = Path("/workspace/outputs/pass8_curate_on_done_STATUS.json")
PY = "/venv/main/bin/python"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kw) -> None:
    STATUS.write_text(json.dumps(kw, indent=2, default=str), encoding="utf-8")


def calibrate_alive() -> list[str]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "calibrate_v4_pass8.py"], text=True
        )
        return [ln for ln in out.splitlines() if "pgrep" not in ln and ln.strip()]
    except subprocess.CalledProcessError:
        return []


def shard_lines() -> dict[str, int]:
    lines = {}
    for s in SHARDS:
        if s.exists() and s.stat().st_size > 0:
            lines[s.name] = sum(1 for _ in s.open(encoding="utf-8", errors="replace"))
        else:
            lines[s.name] = 0
    return lines


def main() -> None:
    log("monitor-only curate-on-done started (no relaunch)")
    write_status(state="waiting")
    empty_dead_ticks = 0
    while True:
        alive = calibrate_alive()
        lines = shard_lines()
        write_status(state="waiting", alive=len(alive), lines=lines)

        if not alive and all(v > 0 for v in lines.values()):
            log(f"shards ready lines={lines} — running curate")
            write_status(state="curating", lines=lines)
            cmd = [
                PY,
                str(CURATE),
                "--candidates",
                str(SHARDS[0]),
                str(SHARDS[1]),
                "--out",
                str(SELECTED),
                "--allow-missing-decontam",
            ]
            proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(proc.stdout[-8000:] + "\n" + proc.stderr[-8000:] + "\n")
            log(f"curate rc={proc.returncode} selected_exists={SELECTED.exists()}")
            if proc.returncode == 0 and SELECTED.exists():
                # soft-promote copy into data/ for GRPO readiness (e2e promote may still run later)
                dest = REPO / "data/arabic_reasoning_rlvr_v5.jsonl"
                dest.write_bytes(SELECTED.read_bytes())
                n = sum(1 for _ in dest.open(encoding="utf-8", errors="replace"))
                log(f"copied selected -> {dest} lines={n}")
                write_status(state="done", lines=lines, selected=n, path=str(dest))
            else:
                write_status(state="curate_failed", rc=proc.returncode, lines=lines)
            return

        if not alive and all(v == 0 for v in lines.values()):
            empty_dead_ticks += 1
            log(
                f"WARNING: calibrate dead, shards empty (tick={empty_dead_ticks}) — NOT relaunching"
            )
            write_status(state="dead_empty", ticks=empty_dead_ticks)
            if empty_dead_ticks >= 10:
                log("giving up after prolonged dead_empty")
                return
            time.sleep(60)
            continue

        empty_dead_ticks = 0
        time.sleep(90)


if __name__ == "__main__":
    main()
