#!/usr/bin/env python3
"""Deploy post-pass@8 waiters: curate (if needed), SFT bench (T06+HF), GRPO chain."""
from __future__ import annotations

from pathlib import Path

import paramiko

HOST = "8.243.214.78"
PORT = 49198
REPO_LOCAL = Path(r"C:\Users\Azooo\arabic-reasoning-rlvr-sota\RL-ar-")
REMOTE_OUT = "/workspace/outputs"

FILES = [
    (REPO_LOCAL / "outputs" / "_pass8_curate_on_done.py", f"{REMOTE_OUT}/_pass8_curate_on_done.py"),
    (REPO_LOCAL / "outputs" / "vast_sft_bench_waiter.py", f"{REMOTE_OUT}/vast_sft_bench_waiter.py"),
    (REPO_LOCAL / "outputs" / "vast_grpo_after_bench_waiter.py", f"{REMOTE_OUT}/vast_grpo_after_bench_waiter.py"),
    (REPO_LOCAL / "outputs" / "vast_pass8_balanced2k_waiter.py", f"{REMOTE_OUT}/vast_pass8_balanced2k_waiter.py"),
    (REPO_LOCAL / "configs" / "qwen_4b_2x5090_v4_sota.yaml", "/workspace/RL-ar-/configs/qwen_4b_2x5090_v4_sota.yaml"),
    (
        REPO_LOCAL / "src" / "rlvr_pipeline" / "hub_checkpoint_callback.py",
        "/workspace/RL-ar-/src/rlvr_pipeline/hub_checkpoint_callback.py",
    ),
]

LAUNCH = r"""
set +e
PY=/venv/main/bin/python
# Never touch active pass@8
PASS8=$(pgrep -af 'calibrate_v4_pass8.py' | grep -v pgrep | wc -l)
echo pass8_pids=$PASS8

start_waiter() {
  name=$1
  script=$2
  log=$3
  if pgrep -f "$script" | grep -qv pgrep; then
    echo "$name already running"
    pgrep -af "$script" | grep -v pgrep | head -1
    return 0
  fi
  nohup $PY -u "$script" >"$log" 2>&1 &
  echo "$name started pid=$!"
  sleep 2
  pgrep -af "$script" | grep -v pgrep | head -1
}

start_waiter curate /workspace/outputs/_pass8_curate_on_done.py /workspace/outputs/pass8_curate_on_done.nohup
start_waiter pass8_balanced2k /workspace/outputs/vast_pass8_balanced2k_waiter.py /workspace/outputs/pass8_balanced2k_waiter.nohup
start_waiter sft_bench /workspace/outputs/vast_sft_bench_waiter.py /workspace/outputs/vast_sft_bench_waiter.nohup
start_waiter grpo_chain /workspace/outputs/vast_grpo_after_bench_waiter.py /workspace/outputs/v5_grpo_after_bench.nohup

echo ===WAITERS===
pgrep -af '_pass8_curate_on_done|vast_pass8_balanced2k|vast_sft_bench_waiter|vast_grpo_after_bench' | grep -v pgrep
echo ===STALE_SUMMARY===
ls -la /workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5/summary* 2>/dev/null | head -5
echo ===RLVR===
wc -l /workspace/RL-ar-/data/arabic_reasoning_rlvr_v5.jsonl 2>/dev/null || echo rlvr_v5_not_yet
"""


def main() -> None:
    key = paramiko.Ed25519Key.from_private_key_file(
        str(Path.home() / ".ssh" / "id_ed25519"), password="aziz"
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", pkey=key, timeout=30, allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()
    for local, remote in FILES:
        if not local.is_file():
            raise FileNotFoundError(local)
        remote_dir = str(Path(remote).parent).replace("\\", "/")
        try:
            sftp.stat(remote_dir)
        except OSError:
            pass
        sftp.put(str(local), remote)
        print(f"uploaded {local.name} -> {remote}")
    sftp.close()
    _, o, e = c.exec_command(LAUNCH, timeout=60)
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("STDERR:", err[:500])
    c.close()


if __name__ == "__main__":
    main()
