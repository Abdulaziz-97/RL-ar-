"""Upload and nohup the durable Vast SFT bench waiter; verify alive."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

LOCAL = Path(__file__).resolve().parent / "vast_sft_bench_waiter.py"
REMOTE = "/workspace/outputs/vast_sft_bench_waiter.py"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = paramiko.Ed25519Key.from_private_key_file(
        str(Path.home() / ".ssh" / "id_ed25519"), password="aziz"
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        "8.243.214.78",
        port=49198,
        username="root",
        pkey=key,
        timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )

    sftp = c.open_sftp()
    sftp.put(str(LOCAL), REMOTE)
    sftp.chmod(REMOTE, 0o755)
    sftp.close()
    print(f"uploaded -> {REMOTE}", flush=True)

    # Use a remote bootstrap script to avoid quoting hell
    bootstrap = r"""#!/bin/bash
set -e
ls -la /workspace/outputs/vast_sft_bench_waiter.py
/venv/main/bin/python -c 'import ast; ast.parse(open("/workspace/outputs/vast_sft_bench_waiter.py").read()); print("syntax_ok")'
pkill -f '/workspace/outputs/vast_sft_bench_waiter.py' 2>/dev/null || true
sleep 1
: > /workspace/outputs/vast_sft_bench_waiter.log
nohup /venv/main/bin/python /workspace/outputs/vast_sft_bench_waiter.py \
  >> /workspace/outputs/vast_sft_bench_waiter.log 2>&1 &
NP=$!
echo NOHUP_PID=$NP
disown "$NP" 2>/dev/null || true
sleep 4
if ps -p "$NP" >/dev/null 2>&1; then
  echo PROCESS_ALIVE=$NP
else
  echo PROCESS_DEAD
fi
pgrep -af vast_sft_bench_waiter.py | grep -v pgrep || echo NO_PGREP
echo ---LOG---
head -n 40 /workspace/outputs/vast_sft_bench_waiter.log || true
echo ---STATUS---
cat /workspace/outputs/sft_v5_aramath_araifeval_STATUS.json 2>/dev/null || echo no_status_yet
echo ---GPU---
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
echo ---PASS8---
pgrep -af calibrate_v4_pass8 | grep -v pgrep || echo none
if test -f /workspace/outputs/sft_v5_aramath_araifeval_waiter.pid; then
  WP=$(cat /workspace/outputs/sft_v5_aramath_araifeval_waiter.pid)
  if kill -0 "$WP" 2>/dev/null; then echo ALIVE_OK pid=$WP; else echo ALIVE_FAIL; fi
else
  echo ALIVE_FAIL missing_pidfile
fi
"""
    sftp = c.open_sftp()
    with sftp.file("/workspace/outputs/_bootstrap_sft_bench_waiter.sh", "w") as f:
        f.write(bootstrap)
    sftp.chmod("/workspace/outputs/_bootstrap_sft_bench_waiter.sh", 0o755)
    sftp.close()

    _, o, e = c.exec_command("bash /workspace/outputs/_bootstrap_sft_bench_waiter.sh", timeout=180)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    print(out, flush=True)
    if err.strip():
        print("STDERR", err[:2000], flush=True)

    time.sleep(2)
    _, o2, _ = c.exec_command(
        "pgrep -af vast_sft_bench_waiter.py | grep -v pgrep; "
        "WP=$(cat /workspace/outputs/sft_v5_aramath_araifeval_waiter.pid 2>/dev/null); "
        "kill -0 \"$WP\" 2>/dev/null && echo CONFIRM_ALIVE=$WP || echo CONFIRM_DEAD",
        timeout=60,
    )
    confirm = o2.read().decode("utf-8", "replace")
    print("===CONFIRM===", confirm, flush=True)
    c.close()
    ok = ("ALIVE_OK" in out) or ("CONFIRM_ALIVE=" in confirm)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
