#!/usr/bin/env python3
"""Kill wrong SFT bench; launch stable HF pass@8 2-shard on Vast."""
from __future__ import annotations

import time
from pathlib import Path

import paramiko

HOST = "8.243.214.78"
PORT = 49198
PASS = "aziz"

REMOTE = r"""
set -euo pipefail
cd /workspace/RL-ar-

echo '===KILL_BAD_BENCH==='
pkill -f 'scripts/run_araeval.py' 2>/dev/null || true
pkill -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -f 'vllm' 2>/dev/null || true
# also kill any stuck calibrate
pkill -f 'calibrate_v4_pass8.py' 2>/dev/null || true
sleep 3
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# free residual if still held
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  if [ -n "$p" ] && [ "$p" != "pid" ]; then
    echo "killing leftover GPU pid $p"
    kill -9 "$p" 2>/dev/null || true
  fi
done
sleep 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

CAND=""
for p in \
  /workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl \
  /workspace/RL-ar-/data_1/outputs/rlvr_candidates/release_corpora/rlvr_train.jsonl \
  /workspace/data/arabic_reasoning_rlvr_candidates_v5.jsonl
do
  if [ -f "$p" ]; then CAND="$p"; break; fi
done
echo "CAND=$CAND lines=$(wc -l < "$CAND")"
SFT=/workspace/outputs/sft_coldstart_v4_v5_20260801_174829
test -f "$SFT/adapter_config.json"
OUT=/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8
mkdir -p "$OUT"
# remove empty/partial failed shards so curate doesn't see stale empties
rm -f "$OUT"/cand_shard*.jsonl "$OUT"/cal_shard*.json 2>/dev/null || true

PY=/venv/main/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf}"
export TOKENIZERS_PARALLELISM=false

# batch=8 prompts * n=8 = 64 seqs — fill more of 5090; fall back to 4 if OOM
BATCH=8
for shard in 0 1; do
  LOG=/workspace/outputs/pass8_hf_stable_shard${shard}.log
  CUDA_VISIBLE_DEVICES=$shard nohup "$PY" data_1/scripts/calibrate_v4_pass8.py \
    --candidates "$CAND" \
    --sft-checkpoint "$SFT" \
    --out-calibration "$OUT/cal_shard${shard}.json" \
    --out-candidates "$OUT/cand_shard${shard}.jsonl" \
    --shard-id "$shard" \
    --num-shards 2 \
    --backend hf \
    --n 8 \
    --temperature 0.7 \
    --max-new-tokens 768 \
    --prompt-batch-size "$BATCH" \
    --merge-sft-lora \
    >"$LOG" 2>&1 &
  echo "launched shard$shard pid=$! log=$LOG"
done

sleep 8
echo '===PROCS==='
pgrep -af 'calibrate_v4_pass8' | grep -v pgrep || echo NONE
echo '===GPU==='
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo '===LOG_HEAD==='
head -n 30 /workspace/outputs/pass8_hf_stable_shard0.log 2>/dev/null | tr -cd '\11\12\15\40-\176\n' | head -30
echo '---'
head -n 20 /workspace/outputs/pass8_hf_stable_shard1.log 2>/dev/null | tr -cd '\11\12\15\40-\176\n' | head -20
"""

def main() -> None:
    key = paramiko.Ed25519Key.from_private_key_file(
        str(Path.home() / ".ssh" / "id_ed25519"), password=PASS
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST, port=PORT, username="root", pkey=key, timeout=40,
        allow_agent=False, look_for_keys=False,
    )
    # long command — run via bash -lc with stdin script to avoid quoting hell
    sftp = c.open_sftp()
    with sftp.file("/tmp/launch_stable_pass8.sh", "w") as f:
        f.write(REMOTE)
    sftp.chmod("/tmp/launch_stable_pass8.sh", 0o755)
    sftp.close()
    _, o, e = c.exec_command("bash /tmp/launch_stable_pass8.sh", timeout=180)
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("STDERR:", err[:2000])
    c.close()


if __name__ == "__main__":
    main()
