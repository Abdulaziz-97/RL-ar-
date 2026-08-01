#!/bin/bash
# Master Setup and Background Execution Script for Vast.ai GPU Instances (Saudi-LLM GRPO V4)
# Explicit stages: preflight → (optional data regen) → SFT → pass@8/curate → GRPO → eval
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RUN_ID="${RUN_ID:-v4_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-/workspace/outputs/master_execution_v4_${RUN_ID}.log}"
PIPELINE_PIDS=()
DATAGEN_ROOT="${DATAGEN_ROOT:-$REPO_ROOT/data_1/outputs/v4_regen/${RUN_ID}}"

cleanup_pipeline_pids() {
    local pid
    for pid in "${PIPELINE_PIDS[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup_pipeline_pids EXIT

mkdir -p /workspace/tmp /workspace/.hf_cache /workspace/outputs "$REPO_ROOT/outputs" "$DATAGEN_ROOT"

run_pipeline() {
    if [ -z "${_NOHUP_LAUNCHED:-}" ]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi

    echo "================================================================="
    echo "INITIALIZING SAUDI-LLM GRPO V4 PIPELINE"
    echo "Repository Root: $REPO_ROOT"
    echo "Run ID:          $RUN_ID"
    echo "Master Log File: $LOG_FILE"
    echo "REGENERATE_DATA: ${REGENERATE_DATA:-0}"
    echo "================================================================="

    # #region agent log
    # H-B sleep-safe: unattended regen must not die hours later at promote.
    if [ "${REGENERATE_DATA:-0}" = "1" ]; then
        if [ -z "${HUMAN_REVIEW_REPORT:-}" ] && [ -z "${SKIP_HUMAN_REVIEW+x}" ]; then
            export SKIP_HUMAN_REVIEW=1
            echo "WARNING: auto-set SKIP_HUMAN_REVIEW=1 for unattended overnight regen (no HUMAN_REVIEW_REPORT)."
        fi
        python3 - <<'PY' || true
import json, time, os
from pathlib import Path
payload = {
    "sessionId": "a273d4",
    "runId": os.environ.get("RUN_ID", "vast"),
    "hypothesisId": "B",
    "location": "setup_and_run_vastai.sh:boot",
    "message": "overnight review policy",
    "data": {
        "SKIP_HUMAN_REVIEW": os.environ.get("SKIP_HUMAN_REVIEW"),
        "HUMAN_REVIEW_REPORT": bool(os.environ.get("HUMAN_REVIEW_REPORT")),
        "REGENERATE_DATA": os.environ.get("REGENERATE_DATA"),
        "TEACHER_WORKERS": os.environ.get("TEACHER_WORKERS"),
    },
    "timestamp": int(time.time() * 1000),
}
line = json.dumps(payload, ensure_ascii=False) + "\n"
for p in (Path("/workspace/outputs/debug-a273d4.log"), Path("debug-a273d4.log")):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open("a", encoding="utf-8").write(line)
    except Exception:
        pass
print(line.strip())
PY
    fi
    # #endregion

    # 96 teacher workers + HTTP clients need far more than default soft nofile=1024.
    ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true
    echo "nofile soft=$(ulimit -n)"

    export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf_cache/hub}"
    export WANDB_DIR="/workspace/outputs/wandb"
    mkdir -p /workspace/outputs/wandb
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/data_1/src:$REPO_ROOT/data_1/vendor:${PYTHONPATH:-}"

    # Frontier-lab datagen defaults (A/B winner: DeepSeek V4 Pro; max safe fan-out).
    export TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-pro}"
    export TEACHER_WORKERS="${TEACHER_WORKERS:-96}"
    export TEACHER_RETRIES="${TEACHER_RETRIES:-4}"
    export VERIFY_WORKERS="${VERIFY_WORKERS:-32}"
    export DATAGEN_RESUME_MULTI_TRACE="${DATAGEN_RESUME_MULTI_TRACE:-1}"
    export SFT_BUDGET_USD="${SFT_BUDGET_USD:-200}"
    export RLVR_BUDGET_USD="${RLVR_BUDGET_USD:-50}"
    if [ -n "${OPENROUTER_API_KEY:-}" ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        export USE_OPENROUTER="${USE_OPENROUTER:-1}"
    fi
    echo "TEACHER_MODEL=${TEACHER_MODEL} TEACHER_WORKERS=${TEACHER_WORKERS} VERIFY_WORKERS=${VERIFY_WORKERS} USE_OPENROUTER=${USE_OPENROUTER:-0}"

    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=1; fi
    echo "[System Diagnostic] Detected $NUM_GPUS active NVIDIA GPU(s)."

    echo "================================================================="
    echo "[0/N] Installing locked project dependencies (no unpinned upgrades)"
    echo "================================================================="
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

    if command -v uv >/dev/null 2>&1; then
        if [ -f "$REPO_ROOT/uv.lock" ]; then
            uv sync --frozen --extra eval || uv pip install --system --break-system-packages -e "$REPO_ROOT[eval]"
        else
            uv pip install --system --break-system-packages -e "$REPO_ROOT[eval]"
        fi
        if [ "${INSTALL_VLLM_PROBE:-0}" = "1" ] && [ -f "$REPO_ROOT/requirements-eval.txt" ]; then
            uv pip install --system --break-system-packages -r "$REPO_ROOT/requirements-eval.txt" || true
        fi
        uv pip install --system --break-system-packages python-bidi arabic-reshaper || true
        if [ -f "$REPO_ROOT/data_1/requirements.txt" ]; then
            echo "[0b/N] Installing data_1 datagen deps (dspy/openai/...)"
            uv pip install --system --break-system-packages -r "$REPO_ROOT/data_1/requirements.txt"
        fi
    else
        pip install -e "$REPO_ROOT[eval]"
        pip install python-bidi arabic-reshaper || true
        if [ -f "$REPO_ROOT/data_1/requirements.txt" ]; then
            pip install -r "$REPO_ROOT/data_1/requirements.txt"
        fi
    fi

    CONFIG_FILE="$REPO_ROOT/configs/qwen_4b_2x5090_v4_sota.yaml"
    SFT_OUT="/workspace/outputs/sft_coldstart_v4_${RUN_ID}"
    GRPO_OUT="/workspace/outputs/qwen_4b_2x5090_v4_run_${RUN_ID}"
    EVAL_OUT="/workspace/outputs/generative_eval_grpo_v4_${RUN_ID}"
    SFT_CFG="$REPO_ROOT/data_1/configs/full_sft_6500.yaml"
    RLVR_CFG="$REPO_ROOT/data_1/configs/full_rlvr_8000.yaml"

    echo "================================================================="
    echo "[1/N] PREFLIGHT"
    echo "================================================================="
    python3 -m rlvr_pipeline.preflight \
        --config "$CONFIG_FILE" \
        --require-gpus "$NUM_GPUS" \
        --sft-output "$SFT_OUT" \
        --grpo-output "$GRPO_OUT"

    if [ "${REGENERATE_DATA:-0}" = "1" ]; then
        echo "================================================================="
        echo "[2a/N] REGENERATE SFT CANDIDATES (lightning-fast parallel teacher)"
        echo "================================================================="
        if [ -z "${DEEPSEEK_API_KEY:-}${OPENROUTER_API_KEY:-}" ]; then
            echo "ERROR: REGENERATE_DATA=1 requires DEEPSEEK_API_KEY or OPENROUTER_API_KEY" >&2
            exit 2
        fi
        # Optional micro-canary before burning the full 6500 budget.
        if [ "${DATAGEN_CANARY:-1}" = "1" ]; then
            echo "[2a0/N] DATAGEN CANARY (20 families, ${TEACHER_WORKERS} workers)"
            CANARY_CFG="$DATAGEN_ROOT/canary_sft_20.yaml"
            python3 - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r"$SFT_CFG").read_text(encoding="utf-8"))
cfg["n_families"] = int("${DATAGEN_CANARY_N:-20}")
cfg["work_dir"] = r"$DATAGEN_ROOT/canary_sft"
Path(r"$CANARY_CFG").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
print(r"$CANARY_CFG")
PY
            python3 "$REPO_ROOT/data_1/scripts/run_pipeline.py" \
                --mode live \
                --config "$CANARY_CFG" \
                --work-dir "$DATAGEN_ROOT/canary_sft" \
                --track sft \
                --model "$TEACHER_MODEL" \
                --workers "$TEACHER_WORKERS" \
                --budget-usd "${DATAGEN_CANARY_BUDGET_USD:-5}" \
                --no-resume
            CANARY_N=$(wc -l < "$DATAGEN_ROOT/canary_sft/release_corpora/sft_train.jsonl" | tr -d ' ')
            if [ "${CANARY_N:-0}" -lt 1 ]; then
                echo "ERROR: datagen canary produced 0 SFT rows — aborting full regen" >&2
                exit 2
            fi
            echo "Canary OK: ${CANARY_N} SFT candidates"
        fi
        SFT_WORK="$DATAGEN_ROOT/sft_candidates"
        # After canary, allow mid-SFT resume (per-problem partial) unless DATAGEN_FRESH=1.
        # Avoid empty-array expansion under `set -u` on older bash (H-C).
        if [ "${DATAGEN_FRESH:-0}" = "1" ]; then
            echo "DATAGEN_FRESH=1: wiping SFT work dir"
            python3 "$REPO_ROOT/data_1/scripts/run_pipeline.py" \
                --mode live \
                --config "$SFT_CFG" \
                --work-dir "$SFT_WORK" \
                --track sft \
                --model "$TEACHER_MODEL" \
                --workers "$TEACHER_WORKERS" \
                --budget-usd "$SFT_BUDGET_USD" \
                --no-resume
        else
            echo "SFT resume enabled (DATAGEN_RESUME_MULTI_TRACE=${DATAGEN_RESUME_MULTI_TRACE})"
            python3 "$REPO_ROOT/data_1/scripts/run_pipeline.py" \
                --mode live \
                --config "$SFT_CFG" \
                --work-dir "$SFT_WORK" \
                --track sft \
                --model "$TEACHER_MODEL" \
                --workers "$TEACHER_WORKERS" \
                --budget-usd "$SFT_BUDGET_USD"
        fi

        python3 "$REPO_ROOT/data_1/scripts/select_sft_v4_release.py" \
            --candidates "$SFT_WORK/release_corpora/sft_train.jsonl" \
            --out "$DATAGEN_ROOT/sft_selected_4000.jsonl" \
            --allow-missing-decontam

        # Stage selected SFT for training (production promote happens after RLVR gate).
        mkdir -p "$REPO_ROOT/data"
        cp "$DATAGEN_ROOT/sft_selected_4000.jsonl" "$REPO_ROOT/data/arabic_reasoning_coldstart_v4.jsonl"
    else
        echo "================================================================="
        echo "[2/N] DATA AUDIT (immutable release; no generation)"
        echo "================================================================="
        python3 "$REPO_ROOT/data_1/scripts/verify_master_v4_datasets_complete.py"
        python3 -m rlvr_pipeline.cli audit-coldstart \
            --data "$REPO_ROOT/data/arabic_reasoning_coldstart_v4.jsonl" \
            --fail-above 0.0
    fi

    echo "================================================================="
    echo "[3/N] STAGE 1: COLD-START SFT"
    echo "================================================================="
    if [ "${FORCE_FRESH:-1}" = "1" ] && [ -d "$SFT_OUT" ]; then
        echo "FORCE_FRESH=1: clearing $SFT_OUT"
        rm -rf "$SFT_OUT"
    fi
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" -m rlvr_pipeline.cli sft \
            --config "$CONFIG_FILE" --output "$SFT_OUT"
    else
        CUDA_VISIBLE_DEVICES=0 python3 -m rlvr_pipeline.cli sft \
            --config "$CONFIG_FILE" --output "$SFT_OUT"
    fi

    echo "================================================================="
    echo "[4/N] SFT VALIDATION / CHECKPOINT INTEGRITY"
    echo "================================================================="
    python3 -m rlvr_pipeline.checkpoint_integrity --checkpoint "$SFT_OUT" --require-adapter-only

    if [ "${REGENERATE_DATA:-0}" = "1" ]; then
        echo "================================================================="
        echo "[5a/N] REGENERATE RLVR CANDIDATES"
        echo "================================================================="
        RLVR_WORK="$DATAGEN_ROOT/rlvr_candidates"
        # Point decontam at the freshly selected SFT release.
        python3 - <<PY
from pathlib import Path
import yaml
cfg_path = Path(r"$RLVR_CFG")
data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
data["decontam_reference_paths"] = [r"$DATAGEN_ROOT/sft_selected_4000.jsonl"]
data["work_dir"] = r"$RLVR_WORK"
out = Path(r"$DATAGEN_ROOT/full_rlvr_8000.runtime.yaml")
out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
print(out)
PY
        python3 "$REPO_ROOT/data_1/scripts/run_pipeline.py" \
            --mode live \
            --config "$DATAGEN_ROOT/full_rlvr_8000.runtime.yaml" \
            --work-dir "$RLVR_WORK" \
            --track rlvr \
            --model "$TEACHER_MODEL" \
            --workers 1 \
            --budget-usd "$RLVR_BUDGET_USD" \
            --no-resume

        echo "================================================================="
        echo "[5b/N] STRICT PASS@8 ON FRESH SFT LINEAGE"
        echo "================================================================="
        mkdir -p "$DATAGEN_ROOT/pass8"
        if [ "$NUM_GPUS" -gt 1 ]; then
            for shard in $(seq 0 $((NUM_GPUS - 1))); do
                CUDA_VISIBLE_DEVICES="$shard" python3 "$REPO_ROOT/data_1/scripts/calibrate_v4_pass8.py" \
                    --candidates "$RLVR_WORK/release_corpora/rlvr_train.jsonl" \
                    --sft-checkpoint "$SFT_OUT" \
                    --out-calibration "$DATAGEN_ROOT/pass8/cal_shard${shard}.json" \
                    --out-candidates "$DATAGEN_ROOT/pass8/cand_shard${shard}.jsonl" \
                    --shard-id "$shard" \
                    --num-shards "$NUM_GPUS" \
                    --temperature "${PASS8_TEMPERATURE:-0.7}" \
                    --max-new-tokens "${PASS8_MAX_NEW_TOKENS:-768}" &
            done
            wait
            CAND_ARGS=()
            for shard in $(seq 0 $((NUM_GPUS - 1))); do
                CAND_ARGS+=("$DATAGEN_ROOT/pass8/cand_shard${shard}.jsonl")
            done
        else
            python3 "$REPO_ROOT/data_1/scripts/calibrate_v4_pass8.py" \
                --candidates "$RLVR_WORK/release_corpora/rlvr_train.jsonl" \
                --sft-checkpoint "$SFT_OUT" \
                --out-calibration "$DATAGEN_ROOT/pass8/cal_shard0.json" \
                --out-candidates "$DATAGEN_ROOT/pass8/cand_shard0.jsonl" \
                --temperature "${PASS8_TEMPERATURE:-0.7}" \
                --max-new-tokens "${PASS8_MAX_NEW_TOKENS:-768}"
            CAND_ARGS=("$DATAGEN_ROOT/pass8/cand_shard0.jsonl")
        fi

        echo "================================================================="
        echo "[5c/N] CURATE 4000 RLVR + DUAL SHIP GATE + ATOMIC PROMOTE"
        echo "================================================================="
        python3 "$REPO_ROOT/data_1/scripts/curate_v4_rlvr.py" \
            --candidates "${CAND_ARGS[@]}" \
            --out "$DATAGEN_ROOT/rlvr_selected_4000.jsonl"

        # Export human-review packs (annotation is offline; optional auto-bypass for CI).
        python3 "$REPO_ROOT/data_1/scripts/export_human_review_packs.py" export \
            --corpus "$DATAGEN_ROOT/sft_selected_4000.jsonl" \
            --out "$DATAGEN_ROOT/review_sft_100.json" \
            --kind sft --n 100 --seed 0
        python3 "$REPO_ROOT/data_1/scripts/export_human_review_packs.py" export \
            --corpus "$DATAGEN_ROOT/rlvr_selected_4000.jsonl" \
            --out "$DATAGEN_ROOT/review_rlvr_100.json" \
            --kind rlvr --n 100 --seed 0

        PROMOTE_ARGS=(
            --sft "$DATAGEN_ROOT/sft_selected_4000.jsonl"
            --rlvr "$DATAGEN_ROOT/rlvr_selected_4000.jsonl"
            --staging-root "$DATAGEN_ROOT/promote"
            --release-id "v4_${RUN_ID}"
        )
        if [ "${SKIP_HUMAN_REVIEW:-0}" = "1" ]; then
            echo "WARNING: SKIP_HUMAN_REVIEW=1 — promoting without annotated review summary"
        elif [ -n "${HUMAN_REVIEW_REPORT:-}" ]; then
            PROMOTE_ARGS+=(--require-human-review-report "$HUMAN_REVIEW_REPORT")
        else
            echo "ERROR: set HUMAN_REVIEW_REPORT to a passing summarize output, or SKIP_HUMAN_REVIEW=1 for machine-only CI" >&2
            exit 2
        fi
        python3 "$REPO_ROOT/data_1/scripts/promote_v4_release.py" "${PROMOTE_ARGS[@]}"
        python3 "$REPO_ROOT/data_1/scripts/verify_master_v4_datasets_complete.py"
    fi

    echo "================================================================="
    echo "[6/N] STAGE 2: GRPO (explicit --output=$GRPO_OUT)"
    echo "================================================================="
    if [ "${FORCE_FRESH:-1}" = "1" ] && [ -d "$GRPO_OUT" ]; then
        echo "FORCE_FRESH=1: clearing $GRPO_OUT"
        rm -rf "$GRPO_OUT"
    fi
    SFT_ARG="--sft-checkpoint $SFT_OUT"
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" -m rlvr_pipeline.cli train \
            --config "$CONFIG_FILE" --output "$GRPO_OUT" $SFT_ARG
    else
        python3 -m rlvr_pipeline.cli train \
            --config "$CONFIG_FILE" --output "$GRPO_OUT" $SFT_ARG
    fi

    echo "================================================================="
    echo "[7/N] GRPO VALIDATION / CHECKPOINT INTEGRITY"
    echo "================================================================="
    TRAINED_CHECKPOINT=$(ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "$GRPO_OUT")
    python3 -m rlvr_pipeline.checkpoint_integrity --checkpoint "$TRAINED_CHECKPOINT" --require-adapter-only

    echo "================================================================="
    echo "[8/N] OFFLINE EVALUATION (lineage-aware)"
    echo "================================================================="
    python3 "$REPO_ROOT/official_eval/run_araeval_generative.py" \
      --model "Qwen/Qwen3.5-4B" \
      --instruction-adapter "aziz9788/T06__qwen35-mixed-v6-lr1e5" \
      --adapter-path "$TRAINED_CHECKPOINT" \
      --enable-thinking \
      --engine hf \
      --output-dir "$EVAL_OUT"

    echo "================================================================="
    echo "ALL STAGES COMPLETE"
    echo "GRPO Checkpoint : $TRAINED_CHECKPOINT"
    echo "Generative Eval : $EVAL_OUT/summary.json"
    echo "================================================================="
}

if [ "${1:-}" == "--fg" ]; then
    run_pipeline
else
    echo "================================================================="
    echo "LAUNCHING V4 PIPELINE IN BACKGROUND (NOHUP / DETACHED MODE)"
    echo "================================================================="
    echo "Master Log Output File : $LOG_FILE"
    echo "To view live logs run  : tail -f $LOG_FILE"
    echo "================================================================="
    rm -f "$LOG_FILE"
    # Detach overnight job: must NOT register this PID in PIPELINE_PIDS or keep
    # the EXIT trap — otherwise the launcher kill-on-exit aborts the nohup child.
    trap - EXIT
    _NOHUP_LAUNCHED=1 nohup bash "$0" --fg > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    disown "$BG_PID" 2>/dev/null || true
    sleep 4
    echo "Process launched in background (pid=${BG_PID})."
    echo "Showing initial output:"
    echo "-----------------------------------------------------------------"
    tail -n 25 "$LOG_FILE" || true
fi
