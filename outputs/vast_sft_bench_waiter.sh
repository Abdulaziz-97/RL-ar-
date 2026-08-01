#!/bin/bash
# Durable on-Vast waiter: poll until pass@8 frees GPUs, then run GRPO_v3-identical
# AraMath + AraIFEval panel on the finished SFT adapter.
set -uo pipefail

ADAPTER="/workspace/outputs/sft_coldstart_v4_v5_20260801_174829"
OUT="/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5"
LOG="/workspace/outputs/sft_v5_aramath_araifeval_panel.log"
STATUS="/workspace/outputs/sft_v5_aramath_araifeval_STATUS.json"
PIDFILE="/workspace/outputs/sft_v5_aramath_araifeval_waiter.pid"
BENCH_PIDFILE="/workspace/outputs/sft_v5_aramath_araifeval.pid"
POLL_S=60
MAX_WAIT_S=21600
MEM_FREE_MIB=2500

echo $$ > "$PIDFILE"
mkdir -p /workspace/outputs "$OUT"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

write_status() {
  local status="$1"
  local extra="${2:-{}}"
  python3 - <<PY
import json
from pathlib import Path
payload = {
  "status": ${status@Q} if False else "$status",
  "ts": "$(ts)",
  "adapter": "$ADAPTER",
  "log": "$LOG",
  "out": "$OUT",
  "waiter_pid": $(cat "$PIDFILE" 2>/dev/null || echo 0),
}
extra = json.loads('''$extra''')
payload.update(extra)
Path("$STATUS").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2)[:2000])
PY
}

# Fix write_status with a cleaner python helper
write_status() {
  local status="$1"
  local extra_json="${2:-{}}"
  STATUS_VAL="$status" EXTRA_JSON="$extra_json" ADAPTER="$ADAPTER" LOG="$LOG" OUT="$OUT" STATUS_PATH="$STATUS" PIDFILE="$PIDFILE" python3 - <<'PY'
import json, os
from pathlib import Path
payload = {
    "status": os.environ["STATUS_VAL"],
    "ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "adapter": os.environ["ADAPTER"],
    "log": os.environ["LOG"],
    "out": os.environ["OUT"],
    "waiter_pid": int(Path(os.environ["PIDFILE"]).read_text().strip() or "0"),
}
try:
    payload.update(json.loads(os.environ["EXTRA_JSON"]))
except Exception:
    pass
Path(os.environ["STATUS_PATH"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"status": payload["status"], "ts": payload["ts"], **{k: payload.get(k) for k in ("gpus_free","pass8","bench","summary") if k in payload}}, ensure_ascii=False))
PY
}

gpus_free() {
  # pass@8 gone AND both GPUs under MEM_FREE_MIB
  if pgrep -f 'calibrate_v4_pass8' >/dev/null 2>&1; then
    return 1
  fi
  python3 - <<PY
import subprocess, sys
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
vals = [int(x.strip()) for x in raw.splitlines() if x.strip()]
sys.exit(0 if vals and all(v < $MEM_FREE_MIB for v in vals) else 1)
PY
}

bench_running() {
  pgrep -f 'run_araeval_generative.py|Saudi-LLM/Eval/scripts/run_araeval.py|scripts/run_araeval.py' >/dev/null 2>&1
}

summary_ready() {
  [[ -f "$OUT/summary.json" ]]
}

launch_bench() {
  export HF_HOME=/workspace/.hf_home
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="/workspace/RL-ar-/src:/workspace/RL-ar-:${PYTHONPATH:-}"

  EVAL_SCRIPT="/workspace/RL-ar-/official_eval/run_araeval_generative.py"
  test -f "$EVAL_SCRIPT"
  test -f "$ADAPTER/adapter_config.json"
  test -n "$(ls "$ADAPTER"/adapter_model.* 2>/dev/null)"

  INSTRUCTION=$(python3 -c "import json; from pathlib import Path; p=Path('$ADAPTER')/'lineage.json'; d=json.loads(p.read_text()) if p.is_file() else {}; print((d.get('instruction_adapter') or 'aziz9788/T06__qwen35-mixed-v6-lr1e5').strip())")

  mkdir -p "$OUT"
  # Fresh panel for SFT (do not resume GRPO_v3 checkpoint)
  rm -f "$OUT/checkpoint.json" "$OUT/summary.json"

  PY=/workspace/RL-ar-/.venv/bin/python
  if [[ ! -x "$PY" ]]; then
    PY=/venv/main/bin/python
  fi
  if [[ ! -x "$PY" ]]; then
    PY=python3
  fi

  {
    echo "===SFT_V5_BENCH_START $(ts)==="
    echo "adapter=$ADAPTER"
    echo "instruction_adapter=$INSTRUCTION"
    echo "tasks=araeval_aramath araeval_ifeval engine=hf"
    echo "lineage=T06-merge + SFT LoRA (run_araeval_generative)"
  } | tee -a "$LOG"

  nohup "$PY" "$EVAL_SCRIPT" \
    --model Qwen/Qwen3.5-4B \
    --instruction-adapter "$INSTRUCTION" \
    --adapter-path "$ADAPTER" \
    --tasks araeval_aramath araeval_ifeval \
    --engine hf \
    --max-lora-rank 128 \
    --batch-size 8 \
    --output-dir "$OUT" \
    >> "$LOG" 2>&1 &
  echo $! > "$BENCH_PIDFILE"
  sleep 4
  if ! kill -0 "$(cat "$BENCH_PIDFILE")" 2>/dev/null; then
    echo "BENCH_LAUNCH_FAILED" | tee -a "$LOG"
    write_status "bench_launch_failed" "{\"log_tail\": $(python3 -c 'import json; from pathlib import Path; t=Path("'"$LOG"'").read_text(errors="replace")[-1500:]; print(json.dumps(t))')}"
    return 1
  fi
  write_status "launched" "{\"bench_pid\": $(cat "$BENCH_PIDFILE"), \"instruction_adapter\": \"$INSTRUCTION\", \"command\": \"$PY $EVAL_SCRIPT --model Qwen/Qwen3.5-4B --instruction-adapter $INSTRUCTION --adapter-path $ADAPTER --tasks araeval_aramath araeval_ifeval --engine hf --output-dir $OUT\"}"
  echo "BENCH_PID=$(cat "$BENCH_PIDFILE")"
  return 0
}

report_done() {
  python3 - <<'PY'
import json
from pathlib import Path
sft = json.loads(Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5/summary.json").read_text())
grpo_path = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_grpo_v3/summary.json")
grpo = json.loads(grpo_path.read_text()) if grpo_path.exists() else {}
raw = sft.get("raw_percent", {})
grow = grpo.get("raw_percent", {})
report = {
  "status": "done",
  "sft_raw": raw,
  "grpo_v3_raw": grow,
  "compare": {
    "aramath": {"sft": raw.get("aramath"), "grpo_v3": grow.get("aramath"), "delta": (None if raw.get("aramath") is None or grow.get("aramath") is None else raw.get("aramath") - grow.get("aramath"))},
    "ifeval_prompt_strict": {"sft": raw.get("ifeval_prompt_strict"), "grpo_v3": grow.get("ifeval_prompt_strict")},
    "ifeval_instruction_strict": {"sft": raw.get("ifeval_instruction_strict"), "grpo_v3": grow.get("ifeval_instruction_strict")},
  },
  "log": "/workspace/outputs/sft_v5_aramath_araifeval_panel.log",
  "out": "/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5",
}
Path("/workspace/outputs/sft_v5_aramath_araifeval_STATUS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
}

echo "[vast-waiter] start pid=$$ $(ts)"
STARTED=$(date +%s)

if summary_ready; then
  echo "[vast-waiter] summary already present"
  report_done
  exit 0
fi

if bench_running; then
  echo "[vast-waiter] bench already running; monitoring"
  write_status "launched_existing" "{}"
else
  write_status "waiting" "{\"note\": \"polling for pass8 GPU free\"}"
fi

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - STARTED))
  if (( ELAPSED > MAX_WAIT_S )); then
    write_status "timeout" "{\"elapsed_s\": $ELAPSED}"
    echo "[vast-waiter] TIMEOUT"
    exit 2
  fi

  if summary_ready; then
    echo "[vast-waiter] DONE"
    report_done
    exit 0
  fi

  PASS8=0
  pgrep -f 'calibrate_v4_pass8' >/dev/null 2>&1 && PASS8=1
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true)
  echo "[probe] elapsed=${ELAPSED}s pass8=$PASS8 bench=$(bench_running && echo 1 || echo 0) gpu=$GPU"

  if bench_running; then
    write_status "running" "{\"elapsed_s\": $ELAPSED, \"gpu\": $(python3 -c 'import json,os; print(json.dumps(os.environ.get("GPU","")))' 2>/dev/null || echo '""')}"
    # keep monitoring
  elif summary_ready; then
    report_done
    exit 0
  else
    if gpus_free; then
      echo "[vast-waiter] GPUs free — launching SFT Aramath+AraIFEval NOW $(ts)"
      if launch_bench; then
        echo "[vast-waiter] launch ok; monitoring until summary"
      else
        echo "[vast-waiter] launch failed; will retry after sleep"
      fi
    else
      write_status "waiting" "{\"elapsed_s\": $ELAPSED, \"pass8\": $PASS8, \"gpu\": \"$GPU\"}"
    fi
  fi

  sleep "$POLL_S"
done
