"""Wait for pass@8 (+curate) to free GPUs, then run GRPO_v3-identical Aramath+AraIFEval on SFT adapter."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko

HOST = "8.243.214.78"
PORT = 49198
USER = "root"
PASS_PHRASE = "aziz"

SFT_ADAPTER = "/workspace/outputs/sft_coldstart_v4_v5_20260801_174829"
OUT_DIR = "/workspace/outputs/sft_v5_aramath_araifeval_20260801"
EVAL_OUT = "/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5"
LOG = "/workspace/outputs/sft_v5_aramath_araifeval_panel.log"
STATUS_LOCAL = Path(__file__).resolve().parent / "SFT_V5_BENCH_STATUS.json"
GRPO_V3_SUMMARY = "/workspace/Saudi-LLM/Eval/outputs/araeval_panel_grpo_v3/summary.json"

# Same harness as successful GRPO_v3 panel (Saudi-LLM Eval / lm-eval+vLLM):
# tasks araeval_aramath + araeval_ifeval, max_lora_rank=128, gpu_mem=0.70, max_batch=16
REMOTE_LAUNCH = r"""
set -euo pipefail
export HF_HOME=/workspace/.hf_home
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/RL-ar-/src:/workspace/RL-ar-:${PYTHONPATH:-}"
mkdir -p /workspace/outputs /workspace/Saudi-LLM/Eval/outputs

ADAPTER="/workspace/outputs/sft_coldstart_v4_v5_20260801_174829"
EVAL_SCRIPT="/workspace/RL-ar-/official_eval/run_araeval_generative.py"
test -f "$EVAL_SCRIPT"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors" || test -n "$(ls "$ADAPTER"/adapter_model.* 2>/dev/null)"
INSTRUCTION=$(python3 -c "import json; from pathlib import Path; p=Path('$ADAPTER')/'lineage.json'; d=json.loads(p.read_text()) if p.is_file() else {}; print((d.get('instruction_adapter') or 'aziz9788/T06__qwen35-mixed-v6-lr1e5').strip())")

OUT="/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5"
LOG="/workspace/outputs/sft_v5_aramath_araifeval_panel.log"
mkdir -p "$OUT"
rm -f "$OUT/checkpoint.json" "$OUT/summary.json"

PY=/workspace/RL-ar-/.venv/bin/python
if [[ ! -x "$PY" ]]; then PY=/venv/main/bin/python; fi
if [[ ! -x "$PY" ]]; then PY=python3; fi

echo "===SFT_V5_BENCH_START $(date -Is)===" | tee -a "$LOG"
echo "adapter=$ADAPTER" | tee -a "$LOG"
echo "instruction_adapter=$INSTRUCTION" | tee -a "$LOG"
echo "tasks=araeval_aramath araeval_ifeval engine=hf" | tee -a "$LOG"

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
echo $! > /workspace/outputs/sft_v5_aramath_araifeval.pid
echo "PID=$(cat /workspace/outputs/sft_v5_aramath_araifeval.pid)"
sleep 3
ps -p "$(cat /workspace/outputs/sft_v5_aramath_araifeval.pid)" -o pid,etime,cmd || true
tail -n 30 "$LOG" || true
"""

PROBE = r"""
python3 - <<'PY'
import json, subprocess
from pathlib import Path

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()

def pgrep(pat):
    try:
        return sh(f"pgrep -af '{pat}' | grep -v pgrep || true")
    except Exception:
        return ""

pass8 = pgrep("calibrate_v4_pass8")
curate = pgrep("curate_v4_rlvr")
bench = pgrep("run_araeval_generative.py") or pgrep("run_araeval.py")
gpu = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
mem_used = []
for line in gpu.splitlines():
    parts = [p.strip() for p in line.split(",")]
    # "0, 10232 MiB, 34 %"
    try:
        mem_used.append(int(parts[1].split()[0]))
    except Exception:
        mem_used.append(99999)

pass8_dir = Path("/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8")
cands = sorted(pass8_dir.glob("cand_shard*.jsonl")) if pass8_dir.exists() else []
cals = sorted(pass8_dir.glob("cal_shard*.json")) if pass8_dir.exists() else []
cand_lines = 0
for p in cands:
    try:
        cand_lines += sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
    except Exception:
        pass

summary = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_sft_v5/summary.json")
grpo = Path("/workspace/Saudi-LLM/Eval/outputs/araeval_panel_grpo_v3/summary.json")
sft_ok = Path("/workspace/outputs/sft_coldstart_v4_v5_20260801_174829/adapter_config.json").is_file()

# GPUs free enough for vLLM at 0.70 util (~22GiB requested) => prefer < 2GiB used and no pass8
gpus_free = (not pass8.strip()) and all(m < 2500 for m in mem_used)
# Prefer curate done too, but allow launch if pass8 done + GPUs free for >1 poll if curate is CPU-only
curate_busy = bool(curate.strip())

out = {
  "sft_adapter_ok": sft_ok,
  "pass8_procs": pass8[:400],
  "curate_procs": curate[:400],
  "bench_procs": bench[:400],
  "gpu": gpu,
  "mem_used_mib": mem_used,
  "gpus_free": gpus_free,
  "curate_busy": curate_busy,
  "pass8_cand_lines": cand_lines,
  "pass8_cal_files": [p.name for p in cals],
  "sft_summary_exists": summary.is_file(),
  "grpo_summary_exists": grpo.is_file(),
}
if summary.is_file():
    out["sft_summary"] = json.loads(summary.read_text(encoding="utf-8"))
if grpo.is_file():
    g = json.loads(grpo.read_text(encoding="utf-8"))
    out["grpo_v3_raw"] = g.get("raw_percent")
print(json.dumps(out, ensure_ascii=False))
PY
"""


def connect() -> paramiko.SSHClient:
    key = paramiko.Ed25519Key.from_private_key_file(
        str(Path.home() / ".ssh" / "id_ed25519"), password=PASS_PHRASE
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST,
        port=PORT,
        username=USER,
        pkey=key,
        timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )
    return c


def exec_cmd(c: paramiko.SSHClient, cmd: str, timeout: int = 180) -> str:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        out += "\nSTDERR:\n" + err[:2000]
    return out


def save_status(payload: dict) -> None:
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    STATUS_LOCAL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    poll_s = 90
    max_wait_s = 6 * 3600
    started = time.time()
    launched = False
    free_streak = 0

    print(f"[bench-waiter] armed; poll={poll_s}s max_wait={max_wait_s}s", flush=True)
    print(
        "[bench-waiter] will launch Saudi-LLM Eval aramath+ifeval on SFT when pass@8 done + GPUs free",
        flush=True,
    )

    while True:
        elapsed = time.time() - started
        if elapsed > max_wait_s:
            save_status({"status": "timeout", "elapsed_s": elapsed})
            print("[bench-waiter] TIMEOUT waiting for GPUs", flush=True)
            return 2

        c = connect()
        try:
            raw = exec_cmd(c, PROBE, timeout=120).strip()
            # last JSON line
            line = [ln for ln in raw.splitlines() if ln.strip().startswith("{")][-1]
            st = json.loads(line)
            print(
                f"[probe] elapsed={int(elapsed)}s gpus_free={st['gpus_free']} "
                f"curate_busy={st['curate_busy']} pass8_cands={st['pass8_cand_lines']} "
                f"mem={st['mem_used_mib']} bench={bool(st['bench_procs'].strip())} "
                f"summary={st['sft_summary_exists']}",
                flush=True,
            )
            save_status({"status": "waiting", "probe": st, "elapsed_s": elapsed, "launched": launched})

            if st.get("sft_summary_exists") and st.get("sft_summary"):
                print("[bench-waiter] BENCH COMPLETE", flush=True)
                print(json.dumps(st["sft_summary"].get("raw_percent"), indent=2), flush=True)
                print("GRPO_v3 raw:", json.dumps(st.get("grpo_v3_raw"), indent=2), flush=True)
                save_status(
                    {
                        "status": "done",
                        "sft_summary": st["sft_summary"],
                        "grpo_v3_raw": st.get("grpo_v3_raw"),
                        "log": LOG,
                        "eval_out": EVAL_OUT,
                        "command": (
                            "python /workspace/RL-ar-/official_eval/run_araeval_generative.py "
                            "--model Qwen/Qwen3.5-4B "
                            "--instruction-adapter aziz9788/T06__qwen35-mixed-v6-lr1e5 "
                            f"--adapter-path {SFT_ADAPTER} "
                            "--tasks araeval_aramath araeval_ifeval --engine hf "
                            "--max-lora-rank 128 --batch-size 8 "
                            f"--output-dir {EVAL_OUT}"
                        ),
                    }
                )
                return 0

            if launched:
                if not st["bench_procs"].strip() and not st["sft_summary_exists"]:
                    print("[bench-waiter] bench process died without summary — check log", flush=True)
                    log_tail = exec_cmd(c, f"tail -n 80 {LOG}", timeout=60)
                    print(log_tail[-4000:], flush=True)
                    save_status({"status": "bench_failed", "log_tail": log_tail[-4000:]})
                    return 3
                # keep waiting for summary
            else:
                # Ready: pass@8 gone + GPUs free. Prefer curate done; allow after 2 free polls if curate still CPU.
                if st["gpus_free"] and st["sft_adapter_ok"]:
                    free_streak += 1
                else:
                    free_streak = 0

                ready = st["gpus_free"] and st["sft_adapter_ok"] and (
                    (not st["curate_busy"]) or free_streak >= 2
                )
                if ready and not st["bench_procs"].strip():
                    print("[bench-waiter] GPUs free — launching SFT Aramath+AraIFEval bench NOW", flush=True)
                    launch_out = exec_cmd(c, REMOTE_LAUNCH, timeout=120)
                    print(launch_out[-3000:], flush=True)
                    launched = True
                    save_status(
                        {
                            "status": "launched",
                            "probe": st,
                            "launch_out": launch_out[-3000:],
                            "log": LOG,
                            "eval_out": EVAL_OUT,
                        }
                    )
        finally:
            c.close()

        time.sleep(poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
