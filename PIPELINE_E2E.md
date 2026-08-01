# Arabic RLVR V5 — End-to-End Pipeline

Run **Qwen3.5-4B** through: T06 instruction adapter → SFT → SFT eval → pass@8 → curate → GRPO.

## Prerequisites

- **GPU host** (Vast or similar): 2× RTX 5090 (or equivalent), Ubuntu, CUDA
- **Hugging Face token** with write access to your Hub repo
- **DeepSeek API key** (optional, for Flash Reverse-QA on new rows)

## Quick start on Vast

```bash
# 1. Clone and enter repo
git clone <this-repo-url> /workspace/RL-ar-
cd /workspace/RL-ar-

# 2. One-shot setup + launch (creates venv, installs deps)
bash scripts/setup_and_run_vastai.sh

# 3. Deploy post-SFT waiters (pass@8 → curate → GRPO)
python outputs/_deploy_v5_post_pass8_waiters.py
```

## Pipeline stages

| Stage | Script / config | Output |
|-------|-----------------|--------|
| SFT | `configs/qwen_4b_2x5090_v4_sota.yaml` | `/workspace/outputs/sft_coldstart_v4_v5_*` |
| SFT eval | `outputs/vast_sft_bench_waiter.py` → `run_araeval_generative.py` + T06 | panel summary JSON |
| pass@8 | `outputs/vast_pass8_balanced2k_waiter.py` | `data_1/outputs/.../pass8_balanced2k/` |
| Curate | `outputs/_pass8_curate_on_done.py` | `data/arabic_reasoning_rlvr_v5.jsonl` |
| GRPO | `outputs/vast_grpo_after_bench_waiter.py` | Hub checkpoints `checkpoint-evaluated-*` |

## Local data prep (before Vast sync)

```powershell
# Compose balanced 2000 rows (500 easy / deferred / medium / hard)
python scripts/local_v5_hard_med_compose_4k.py

# Flash Reverse-QA on programmatic rows (needs DEEPSEEK_API_KEY)
python scripts/local_v5_rqa_then_pass8.py

# Force RQA on any rows that kept original prompts
python scripts/local_v5_force_rqa_no_fallback.py
```

Sync to Vast:

```bash
scp -P <port> data/arabic_reasoning_rlvr_v5.jsonl data/arabic_reasoning_rlvr_candidates_v5.jsonl root@<host>:/workspace/RL-ar-/data/
```

## GRPO hold (pass@8 gate)

While pass@8 on the balanced 2k set is running, GRPO is blocked:

```bash
touch /workspace/outputs/GRPO_HOLD_UNTIL_PASS8
```

Remove after pass@8 + curate complete:

```bash
rm /workspace/outputs/GRPO_HOLD_UNTIL_PASS8
python outputs/vast_grpo_after_bench_waiter.py   # or restart via deploy script
```

## Key paths on Vast

| Path | Purpose |
|------|---------|
| `/workspace/RL-ar-/data/arabic_reasoning_rlvr_v5.jsonl` | Final RLVR training set |
| `/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl` | pass@8 input |
| `/workspace/outputs/sft_coldstart_v4_v5_20260801_174829` | SFT LoRA checkpoint |
| `/workspace/outputs/pass8_balanced2k_STATUS.json` | pass@8 waiter state |
| `/workspace/outputs/v5_grpo_train.pid` | GRPO process |

## Environment variables

```bash
export HF_TOKEN=...
export DEEPSEEK_API_KEY=...   # local RQA only
export PASS8_OUT=/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8_balanced2k
export CURATE_OUT=/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/rlvr_selected_2000.jsonl
```

## Important notes

1. **SFT eval** must use `run_araeval_generative.py` with T06 merge — not bare `run_araeval.py`.
2. **pass@8** uses `--merge-sft-lora` and `--backend hf` with 2 shards (one GPU each).
3. **GRPO** config: `eval_steps: 62`, Hub prefix `checkpoint-evaluated`.
4. Heuristic difficulty bands on programmatic rows are replaced by empirical pass@8 bands after calibration.

## Repo layout (runtime only)

```
configs/                    # GRPO / training YAML
src/rlvr_pipeline/          # Core pipeline package
data_1/scripts/             # pass@8, curate, RQA helpers
scripts/                    # Local compose + Vast setup
outputs/vast_*.py           # Vast waiters
outputs/_pass8_curate_on_done.py
outputs/_deploy_v5_post_pass8_waiters.py
outputs/_launch_stable_pass8.py
```

Data artifacts (`*.jsonl`, logs, `data_1/outputs/`) are gitignored — generate locally or on Vast.
