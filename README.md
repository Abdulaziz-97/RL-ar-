# Arabic Reasoning RLVR — Team Pack

Self-contained folder to run the full local pipeline: **SFT cold-start → GRPO (QLoRA)**.

Everything lives under `src/` in this folder:

| Package | Role |
|---------|------|
| `rlvr` | Reward math, curriculum, failure mining, monitoring |
| `rlvr_pipeline` | Trainers, configs, CLI, Arabic reward wrappers |
| `rlvr_contracts` | Shared parse / verify contracts (data + rewards) |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (8GB+ VRAM for Qwen3.5-2B QLoRA; more is better)
- Hugging Face access to download `Qwen/Qwen3.5-2B`

## Setup

```bash
cd team_pack
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# Install PyTorch with CUDA 12.4 first
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e .
```

## Data

| Role | File |
|------|------|
| SFT train | `data/arabic_reasoning_coldstart_train.jsonl` |
| SFT eval | `data/arabic_reasoning_coldstart_eval.jsonl` |
| GRPO train | `data/arabic_reasoning_rlvr_train.jsonl` |
| GRPO eval | `data/arabic_reasoning_rlvr_eval.jsonl` |

Corpus: `v2_dspy_gepa`. See `docs/data_v2_dspy_gepa/` for quality notes and ship gate.

## Run the pipeline

### Option A — one script (SFT then GRPO)

```bash
python scripts/run_full_pipeline.py
```

Default: 2 SFT epochs → GRPO smoke with `configs/qwen_4b_smoke_v11.yaml` (30 steps).

### Option B — step by step

```bash
# 1) Cold-start SFT
python -m rlvr_pipeline sft \
  --config configs/qwen_4b_qlora.yaml \
  --output ./runs/sft_v1 \
  --num-train-epochs 2

# 2) Format probe (should pass after SFT)
python scripts/format_probe_sft.py ./runs/sft_v1 512

# 3) GRPO — recommended smoke recipe
python -m rlvr_pipeline train \
  --config configs/qwen_4b_smoke_v11.yaml \
  --sft-checkpoint ./runs/sft_v1 \
  --output ./runs/grpo_v1 \
  --max-steps 30
```

**Do not resume GRPO from a collapsed-entropy checkpoint.** Always start GRPO from a fresh SFT adapter.

## Configs

| File | Use |
|------|-----|
| `configs/qwen_4b_qlora.yaml` | Default QLoRA settings |
| `configs/qwen_4b_smoke_v11.yaml` | Recommended GRPO smoke (post chat-template + adapter fixes) |

## Tests

```bash
pytest tests -q --ignore=tests/rlvr
# Optional: math-layer unit tests
pytest tests/rlvr -q
```

## Important fixes included

1. **Chat template** — strips Qwen3.5 empty `<think></think>` injection so format reward is earnable.
2. **Strict SFT adapter load** — remaps `language_model.*` keys; fails hard on mismatch (no silent base-model training).

## Docs

- `DOCS.md` — full pipeline guide
- `docs/FULL_SCALE_SYNTHETIC_DATA_PROPOSAL.md` — next data build proposal
- `docs/data_v2_dspy_gepa/` — current data handoff / quality

## Layout

```
team_pack/
  configs/
  data/
  docs/
  scripts/
  src/rlvr/
  src/rlvr_pipeline/
  src/rlvr_contracts/
  tests/
  requirements.txt
  pyproject.toml
  README.md
  DOCS.md
```
