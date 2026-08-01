#!/bin/bash
# Master Setup and Background Execution Script for Vast.ai GPU Instances (Saudi-LLM GRPO V4)
# Explicit stages: preflight → data audit → SFT → SFT validate → GRPO → GRPO validate → offline-eval
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RUN_ID="${RUN_ID:-v4_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-/workspace/outputs/master_execution_v4_${RUN_ID}.log}"
PIPELINE_PIDS=()

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

mkdir -p /workspace/tmp /workspace/.hf_cache /workspace/outputs "$REPO_ROOT/outputs"

run_pipeline() {
    if [ -z "${_NOHUP_LAUNCHED:-}" ]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi

    echo "================================================================="
    echo "INITIALIZING SAUDI-LLM GRPO V4 PIPELINE"
    echo "Repository Root: $REPO_ROOT"
    echo "Run ID:          $RUN_ID"
    echo "Master Log File: $LOG_FILE"
    echo "================================================================="

    export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf_cache/hub}"
    export WANDB_DIR="/workspace/outputs/wandb"
    mkdir -p /workspace/outputs/wandb
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=1; fi
    echo "[System Diagnostic] Detected $NUM_GPUS active NVIDIA GPU(s)."

    echo "================================================================="
    echo "[0/7] Installing locked project dependencies (no unpinned upgrades)"
    echo "================================================================="
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

    if command -v uv >/dev/null 2>&1; then
        if [ -f "$REPO_ROOT/uv.lock" ]; then
            uv sync --frozen --extra eval || uv pip install --system --break-system-packages -e "$REPO_ROOT[eval]"
        else
            uv pip install --system --break-system-packages -e "$REPO_ROOT[eval]"
        fi
        # Optional probe deps (vLLM) in a separate requirements file — HF eval is default.
        if [ "${INSTALL_VLLM_PROBE:-0}" = "1" ] && [ -f "$REPO_ROOT/requirements-eval.txt" ]; then
            uv pip install --system --break-system-packages -r "$REPO_ROOT/requirements-eval.txt" || true
        fi
        # Arabic terminal rendering helpers (optional)
        uv pip install --system --break-system-packages python-bidi arabic-reshaper || true
    else
        pip install -e "$REPO_ROOT[eval]"
        pip install python-bidi arabic-reshaper || true
    fi

    CONFIG_FILE="$REPO_ROOT/configs/qwen_4b_2x5090_v4_sota.yaml"
    SFT_OUT="/workspace/outputs/sft_coldstart_v4_${RUN_ID}"
    GRPO_OUT="/workspace/outputs/qwen_4b_2x5090_v4_run_${RUN_ID}"
    EVAL_OUT="/workspace/outputs/generative_eval_grpo_v4_${RUN_ID}"

    echo "================================================================="
    echo "[1/7] PREFLIGHT"
    echo "================================================================="
    python3 -m rlvr_pipeline.preflight \
        --config "$CONFIG_FILE" \
        --require-gpus "$NUM_GPUS" \
        --sft-output "$SFT_OUT" \
        --grpo-output "$GRPO_OUT"

    echo "================================================================="
    echo "[2/7] DATA AUDIT (ship gate + coldstart)"
    echo "================================================================="
    if [ -n "${OPENROUTER_API_KEY:-}" ]; then
        python3 "$REPO_ROOT/data_1/scripts/generate_qwen37_rlvr_live.py" || true
    else
        echo "OPENROUTER_API_KEY not set. Using shipped V4 datasets."
    fi
    python3 "$REPO_ROOT/data_1/scripts/verify_master_v4_datasets_complete.py"
    python3 -m rlvr_pipeline.cli audit-coldstart \
        --data "$REPO_ROOT/data/arabic_reasoning_coldstart_v4.jsonl" \
        --fail-above 0.05

    echo "================================================================="
    echo "[3/7] STAGE 1: COLD-START SFT"
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
    echo "[4/7] SFT VALIDATION / CHECKPOINT INTEGRITY"
    echo "================================================================="
    python3 -m rlvr_pipeline.checkpoint_integrity --checkpoint "$SFT_OUT" --require-adapter-only

    echo "================================================================="
    echo "[5/7] STAGE 2: GRPO (explicit --output=$GRPO_OUT)"
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
    echo "[6/7] GRPO VALIDATION / CHECKPOINT INTEGRITY"
    echo "================================================================="
    TRAINED_CHECKPOINT=$(ls -d "$GRPO_OUT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "$GRPO_OUT")
    python3 -m rlvr_pipeline.checkpoint_integrity --checkpoint "$TRAINED_CHECKPOINT" --require-adapter-only

    echo "================================================================="
    echo "[7/7] OFFLINE EVALUATION (lineage-aware)"
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
    _NOHUP_LAUNCHED=1 nohup bash "$0" --fg > "$LOG_FILE" 2>&1 &
    PIPELINE_PIDS+=($!)
    sleep 4
    echo "Process launched in background (pid=${PIPELINE_PIDS[0]})."
    echo "Showing initial output:"
    echo "-----------------------------------------------------------------"
    tail -n 25 "$LOG_FILE" || true
fi
