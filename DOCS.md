# Arabic Reasoning RLVR Pipeline — Complete Guide

## Table of Contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Where things live (two codebases)](#2-where-things-live-two-codebases)
3. [Architecture deep dive](#3-architecture-deep-dive)
4. [Setup from scratch](#4-setup-from-scratch)
5. [The YAML config — every knob explained](#5-the-yaml-config--every-knob-explained)
6. [Data format — plug in your dataset](#6-data-format--plug-in-your-dataset)
7. [Running locally (RTX 2080 Super 8GB)](#7-running-locally-rtx-2080-super-8gb)
8. [Running on cloud GPU (A100/H100)](#8-running-on-cloud-gpu-a100h100)
9. [The reward system — how the model is scored](#9-the-reward-system--how-the-model-is-scored)
10. [GRPO vs GSPO vs GSPO-token](#10-grpo-vs-gspo-vs-gspo-token)
11. [Monitoring and debugging](#11-monitoring-and-debugging)
12. [The test suite](#12-the-test-suite)
13. [Troubleshooting](#13-troubleshooting)
14. [Extending the pipeline](#14-extending-the-pipeline)

---

## 1. What this pipeline does

It takes a base Arabic language model (Qwen 2.5) and post-trains it to **reason in Arabic** using Reinforcement Learning with Verifiable Rewards (RLVR).

The training loop:

```
Your JSONL data
    →
Load prompts (Arabic math/logic questions)
    →
Model generates G=8 completions per prompt (with <think> and <answer> tags)
    →
Each completion is scored by 5 reward functions:
  correctness (60%) + format (20%) + Arabic language (20%)
  - answer-leak penalty (50%) - structural-leak penalty (30%)
    →
Group-relative advantages computed: A_i = (R_i - mean) / std
    →
GRPO/GSPO loss computed with clipping + KL penalty to reference model
    →
Backpropagate through QLoRA adapters
    →
Repeat until epochs done
```

The model learns to:
- Produce step-by-step reasoning in Arabic inside `<think>` tags
- Output correct final answers inside `<answer>` tags
- Avoid reward hacks (empty reasoning, English reasoning, answer leaks)

## 2. Where things live (two codebases)

There are **two** codebases, not one. They work together.

### Codebase 1: `arabic-reasoning-rlvr` (the "math" layer)

```
C:\Users\Azooo\arabic-reasoning-rlvr\
  src/rlvr/
    config.py               -- PipelineConfig dataclass
    dataset_loader.py       -- Module 1: JSON/JSONL loading + schema validation
    rollout_engine.py       -- Module 2: generate_group() stub (tested with mocks)
    reward_composer.py      -- Module 3: 5 reward scoring functions
    difficulty_labeler.py   -- Module 4: error-rate quartile labeling
    advantage_calculator.py -- Module 5: GRPO advantage formula
    failure_bank.py         -- Module 6: zero-reward group banking
    entropy_guard.py        -- Module 7: Clip-Cov / KL-Cov (pure Python)
    policy_update.py        -- Module 8: GRPO/GSPO loss (PyTorch)
    monitoring.py           -- dashboard metrics computation
    pipeline.py             -- orchestrator tying all 8 modules
  tests/                    -- 100 unit tests
```

**Role**: The algorithmic brain. All reward math, advantage computation, entropy guard logic lives here. It is fully unit-tested with mocked data (100 tests). It has **zero model integration** — pure functions operating on strings and numbers.

### Codebase 2: `arabic-reasoning-rlvr-sota` (the "training" layer)

```
C:\Users\Azooo\arabic-reasoning-rlvr-sota\
  src/rlvr_sota/
    config.py      -- SOTAConfig → TRL GRPOConfig + QLoRA builders
    rewards.py     -- 5 TRL-compatible reward functions (wraps Codebase 1)
    data.py        -- JSONL → HuggingFace Dataset (chat-template formatting)
    trainer.py     -- build_trainer() assembles GRPOTrainer
    cli.py         -- python -m rlvr_sota train --config ...
  configs/
    qwen_4b_qlora.yaml  -- 8GB local config
    cloud_a100.yaml     -- 80GB cloud config
  tests/           -- 39 tests (6 full training integration tests)
```

**Role**: The training engine. Wraps TRL's `GRPOTrainer` (the SOTA open-source GRPO implementation). Handles model loading, QLoRA, generation, batching, optimizer, checkpointing, wandb logging. **Imports Codebase 1** for reward computation via editable pip install.

### How they connect

```
rlvr_sota (Codebase 2)
  imports →
    rlvr (Codebase 1) — for compose_reward, reward_format, etc.
  wraps →
    TRL GRPOTrainer — for the actual training loop
```

Codebase 2 installed Codebase 1 in editable mode:
```bash
pip install -e ../arabic-reasoning-rlvr
```

## 3. Architecture deep dive

```
┌─────────────────────────────────────────────────────────────┐
│                     rlvr_sota (SOTA layer)                  │
│                                                             │
│  CLI: python -m rlvr_sota train --config config.yaml        │
│    │                                                        │
│    ▼                                                        │
│  SOTAConfig.from_yaml() ─── reads YAML, builds GRPOConfig   │
│    │                                                        │
│    ▼                                                        │
│  build_trainer()                                            │
│    ├── Loads model (Qwen 2.5 3B) with bitsandbytes 4-bit    │
│    ├── Applies LoRA adapters (r=64, alpha=128)              │
│    ├── Loads dataset from JSONL as HF conversational format │
│    ├── Wires 5 reward functions from rlvr.reward_composer   │
│    └── Creates TRL GRPOTrainer(model, config, rewards, ds)  │
│         │                                                   │
│         ▼                                                   │
│  trainer.train() ─── the GRPO loop:                         │
│    ┌──────────────────────────────────────────────┐         │
│    │ 1. Sample batch of prompts from dataset       │         │
│    │ 2. Generate G=8 completions per prompt        │         │
│    │    (continuous batching, temp=0.9)            │         │
│    │ 3. Decode → score with 5 reward functions     │
│    │    correct(0.6) + format(0.2) + lang(0.2)    │         │
│    │    - leak(0.5) - struct(0.3)                  │         │
│    │ 4. Compute group advantages: (R_i-mean)/std   │         │
│    │ 5. Compute policy & ref log-probs per token   │         │
│    │ 6. Dr. GRPO loss: min(ratio*adv, clipped)     │         │
│    │    + KL penalty to reference model            │         │
│    │ 7. Backprop (paged_adamw_8bit), grad accum    │         │
│    │ 8. Log to wandb, save checkpoint              │         │
│    └──────────────────────────────────────────────┘         │
│                                                             │
│  Key SOTA features:                                         │
│    • Dr. GRPO loss (removes response-length bias)           │
│    • Batch-scale rewards (local mean + global std)          │
│    • Asymmetric clipping (ε=0.2, ε_high=0.28 — DAPO)       │
│    • KL to reference model via adapter disabling            │
│    • Continuous batching generation (Windows-compatible)    │
│    • GRPO/GSPO/GSPO-token switchable via one config flag    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ imports
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      rlvr (math layer)                      │
│                                                             │
│  reward_composer.py                                         │
│    compose_reward(completion, ground_truth, domain) → 0..1  │
│      = 0.6 * correctness                                    │
│      + 0.2 * format       (must have <think>/<answer>)     │
│      + 0.2 * language     (Arabic word ratio)               │
│      - 0.5 * answer_leak  (direct answer in <think>)        │
│      - 0.3 * struct_leak  (text outside tags)               │
│                                                             │
│  advantage_calculator.py                                    │
│    A_i = (R_i - mean) / (std + 1e-4)  — NaN-safe           │
│                                                             │
│  failure_bank.py                                            │
│    Banks groups where all rewards=0 (replay in hard stage)  │
│                                                             │
│  dataset_loader.py                                          │
│    Validates JSONL schema (domain, puzzle_type, tags)       │
│    Detects near-duplicate prompts (3-gram Jaccard > 0.8)   │
└─────────────────────────────────────────────────────────────┘
```

## 4. Setup from scratch

### Prerequisites
- Windows with PowerShell 5.1
- Python 3.12
- NVIDIA RTX 2080 Super (8GB) or better
- CUDA 12.4
- Git

### Step-by-step

```powershell
# 1. Clone or navigate to the project directories
#    (These should already exist at the paths below)

# 2. Create and activate the SOTA venv
& "C:\Users\Azooo\AppData\Local\Programs\Python\Python312\python.exe" `
  -m venv C:\Users\Azooo\arabic-reasoning-rlvr-sota\.venv

C:\Users\Azooo\arabic-reasoning-rlvr-sota\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install the SOTA pipeline dependencies
pip install trl peft bitsandbytes accelerate wandb pyyaml pytest pytest-cov

# 5. Install the math layer (Codebase 1) in editable mode
pip install -e C:\Users\Azooo\arabic-reasoning-rlvr

# 6. Verify everything works
cd C:\Users\Azooo\arabic-reasoning-rlvr-sota
pytest -q
# Expected: 39 passed
```

### Verify CUDA is available

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True, Device: NVIDIA GeForce RTX 2080 Super
```

## 5. The YAML config — every knob explained

The config file drives everything. Here is every parameter with its meaning and recommended value.

### Model

```yaml
model_name: "Qwen/Qwen2.5-3B-Instruct"  # HF model ID or local path
load_in_4bit: true                        # Use 4-bit NF4 quantization (8GB VRAM)
bnb_4bit_compute_dtype: "bfloat16"        # Computation precision
bnb_4bit_quant_type: "nf4"               # Quantization type (nf4 = best quality)
bnb_4bit_use_double_quant: true          # Double quantization (saves ~0.4GB)
```

**When to change**: 
- If you have 16GB+ VRAM, set `load_in_4bit: false` for higher quality.
- If using a different model family, check which target_modules your model uses (see LoRA below).

### LoRA / QLoRA

```yaml
lora_r: 64             # Rank of LoRA adapters (higher = more capacity, more memory)
lora_alpha: 128        # Scaling factor (alpha/r = effective learning rate multiplier)
lora_dropout: 0.05     # Dropout on LoRA layers (regularization)
```

**When to change**:
- `r=32` if memory is very tight, `r=128` if you have headroom.
- `alpha` should be 2× `r` for standard scaling.

### Data

```yaml
train_data_path: "data/math.jsonl"    # Path to your JSONL training data
eval_data_path: ""                     # Optional eval set (same format)
system_prompt: "..."                   # Arabic system prompt (shown to model)
```

The default system prompt is:
```
أنت مساعد ذكي تحل المسائل خطوة بخطوة. اكتب تفكيرك داخل وسوم <think> والإجابة النهائية داخل وسوم <answer>. استخدم اللغة العربية في التفكير.
```
(You are a smart assistant that solves problems step by step. Write your thinking inside `<think>` tags and the final answer inside `<answer>` tags. Use Arabic in your reasoning.)

### Generation (group sampling)

```yaml
num_generations: 8       # Group size G (number of completions per prompt)
max_completion_length: 1024  # Max new tokens to generate per completion
temperature: 0.9          # Sampling temperature (higher = more diverse)
top_p: 1.0                # Nucleus sampling cutoff
```

**When to change**:
- `num_generations` (G): The GRPO paper uses 4-16. More G = better advantage estimates, but slower. 8 is a good default.
- `max_completion_length`: For Arabic reasoning with `<think>` + `<answer>`, 1024 is generous. Reduce to 512 for simple math, increase to 2048 for long logic problems.
- `temperature`: 0.9 encourages exploration. Lower (0.7) for more deterministic outputs. Higher (1.0+) for even more diversity.

### Training

```yaml
learning_rate: 1.0e-5             # Learning rate for LoRA (lower than full FT)
num_train_epochs: 1               # Number of passes through the dataset
per_device_train_batch_size: 2    # Prompts per GPU per step (4 for 8GB was assumed; use 2 for safety)
gradient_accumulation_steps: 8    # Effective batch = 2 * 8 = 16 prompts per update
max_grad_norm: 1.0                # Gradient clipping
warmup_ratio: 0.1                 # 10% of steps as warmup
lr_scheduler_type: "cosine"       # Cosine decay from peak to 0
optim: "paged_adamw_8bit"         # 8-bit optimizer for memory efficiency
seed: 42                          # Reproducibility seed
```

**When to change**:
- If you hit OOM: reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` to maintain effective batch size.
- Effective batch size = `per_device_train_batch_size × num_generations × gradient_accumulation_steps`. For GRPO, effective batch should be ≥ 128 tokens.
- `learning_rate`: For LoRA, 1e-5 to 5e-5 is typical. Full fine-tuning uses 1e-6.

### Algorithm — this is where SOTA lives

```yaml
loss_type: "dr_grpo"              # Dr. GRPO: removes response-length bias
scale_rewards: "batch"            # Local mean + global std (more robust)
beta: 0.04                        # KL penalty coefficient
epsilon: 0.2                      # PPO clip lower bound
epsilon_high: 0.28                # PPO clip upper bound (asymmetric, DAPO-style)
importance_sampling_level: "token"  # "token"=GRPO, "sequence"=GSPO, "sequence_token"=GSPO-token
mask_truncated_completions: true  # Ignore truncated (cut-off) completions
```

**Loss types explained**:
| `loss_type` | What it does | When to use |
|---|---|---|
| `"grpo"` | Original GRPO, divides by per-response length | Baseline |
| `"dapo"` | Token-level normalization, reduces length bias | Better than grpo for long-CoT tasks |
| `"dr_grpo"` | Constant normalization (max_completion_length), fully removes length bias | **SOTA 2026 default** — use this |
| `"sapo"` | Soft gating instead of hard clipping (Qwen team) | Experimental, may be more stable |
| `"bnpo"` | Batch-normalized policy optimization | Alternative normalization scheme |

**Scale rewards explained**:
| `scale_rewards` | What it does | When to use |
|---|---|---|
| `"group"` (default) | Mean and std computed per-group (each prompt's G completions) | Original GRPO, per-prompt normalization |
| `"batch"` | Mean per-group, std across entire batch | **SOTA 2026** — removes question-difficulty bias |
| `"off"` | No scaling, use raw rewards | Experimental, only if rewards are well-calibrated |

**Importance sampling level**:
| Setting | Algorithm | What changes |
|---|---|---|
| `"token"` | GRPO | Per-token ratios, per-token clipping |
| `"sequence"` | GSPO | Sequence-level ratio (product of token ratios), sequence-level clipping |
| `"sequence_token"` | GSPO-token | Sequence-level ratio with token-level clipping (`trl.experimental.gspo_token`) |

**When to change**:
- GRPO (token): Best for tasks where individual tokens' contributions matter.
- GSPO (sequence): Better when the entire response is one unit (logic puzzles, full derivations).
- GSPO-token: Newest variant, may give best of both. Still experimental.

### Reward weights

```yaml
reward_weights: [0.6, 0.2, 0.2, 0.5, 0.3]
#                 ^    ^    ^    ^    ^
#                 |    |    |    |    structural_leak_penalty  (subtracted)
#                 |    |    |    answer_leak_penalty           (subtracted)
#                 |    |    language_consistency
#                 |    format_compliance
#                 correctness
```

The total reward = weighted sum:
```
0.6 × correctness + 0.2 × format + 0.2 × language
- 0.5 × answer_leak - 0.3 × structural_leak
```

**When to change**: 
- If the model overfits on correctness at the expense of format, increase format weight.
- If you see answer-leak hacks, increase the answer-leak penalty weight.
- Each component is logged separately in wandb, so you can see which signals dominate.

### Infrastructure

```yaml
output_dir: "./outputs"            # Where checkpoints and logs go
logging_steps: 10                  # Log metrics every N steps
save_steps: 500                    # Save checkpoint every N steps
eval_steps: 100                    # Run eval every N steps (optional)
bf16: true                         # bfloat16 mixed precision (saves memory)
gradient_checkpointing: true       # Recompute activations (saves memory)
use_transformers_continuous_batching: true  # Faster generation on single GPU
use_wandb: false                    # Set true to enable wandb dashboard
wandb_project: "arabic-reasoning-rlvr"

log_completions: true              # Print sample completions in logs
num_completions_to_print: 4        # How many completions to show
```

## 6. Data format — plug in your dataset

The pipeline expects a **JSONL file** (one JSON object per line). Each line must follow this schema:

### Math domain

```jsonl
{"id": "math-001", "domain": "math", "difficulty_tag": "easy", "prompt": "ما هو ناتج 2 + 3؟", "ground_truth_answer": 5}
{"id": "math-002", "domain": "math", "difficulty_tag": "easy", "prompt": "احسب 7 × 8؟", "ground_truth_answer": 56}
{"id": "math-003", "domain": "math", "difficulty_tag": "medium", "prompt": "إذا كان x + 5 = 12 فما هو x؟", "ground_truth_answer": 7}
```

### Logic domain

```jsonl
{"id": "logic-001", "domain": "logic", "difficulty_tag": "hard", "prompt": "ثلاثة أشخاص A و B و C يقفون في صف. A ليس في المنتصف. B على يمين C. رتبهم.", "ground_truth_answer": {"A": 1, "B": 3, "C": 2}, "puzzle_type": "ordering"}
```

### Required fields

| Field | Type | Required For | Description |
|---|---|---|---|
| `id` | string | All | Unique identifier for this sample |
| `domain` | string | All | `"math"` or `"logic"` |
| `difficulty_tag` | string | All | Coarse difficulty: `"easy"`, `"medium"`, `"hard"` |
| `prompt` | string | All | The Arabic question/prompt |
| `ground_truth_answer` | any | All | The expected answer. For math: a number (int/float). For logic: a dict/list representing the solution |
| `puzzle_type` | string | Logic only | Type of logic puzzle: `"ordering"`, `"constraint"`, `"grouping"`, etc. |

### How it flows into the model

Internally, each sample is converted into a conversational format before being fed to the model:

```python
# Your JSONL: {"prompt": "ما هو ناتج 2+3؟", "ground_truth_answer": 5, ...}
# Becomes:
[
    {"role": "system", "content": "أنت مساعد ذكي..."},
    {"role": "user",   "content": "ما هو ناتج 2+3؟"}
]
# Model generates (G times with temperature):
# "<think>أجمع العددين معا ثم أحسب الناتج بدقة</think><answer>5</answer>"
# The completion is decoded → 5 reward functions score it → advantage → loss
```

### Validation

The data loader validates:
- All required fields present
- `domain` is exactly `"math"` or `"logic"`
- Logic samples have `puzzle_type`
- No duplicate IDs within a file
- Cold-start CoT data (if used for SFT warm-up) must have `<think>` and `<answer>` tags

### Near-duplicate detection

The `rlvr.dataset_loader` module includes `find_near_duplicate_prompts()` which uses 3-gram shingling + Jaccard similarity to detect prompts in your cold-start and RLVR datasets that are >80% similar. Run this before training to avoid data leakage:

```python
from rlvr.dataset_loader import load_cold_start_dataset, load_rlvr_dataset, find_near_duplicate_prompts

cold = load_cold_start_dataset("cold_start.jsonl")
math = load_rlvr_dataset("math_rlvr.jsonl", "math")

dupes = find_near_duplicate_prompts(cold, math, threshold=0.8)
for cs_id, rl_id in dupes:
    print(f"Duplicate: cold-start {cs_id} ↔ rlvr {rl_id}")
```

## 7. Running locally (RTX 2080 Super 8GB)

### Before you run

The RTX 2080 Super has 8GB VRAM. This config uses aggressive memory optimization:

| Technique | Memory savings |
|---|---|
| 4-bit NF4 quantization | Model: ~2.5GB instead of ~15GB (fp16) |
| LoRA (r=64) instead of full fine-tuning | Only train ~0.5% of params |
| paged_adamw_8bit optimizer | Optimizer states: ~0.3GB instead of ~1.2GB |
| Gradient checkpointing | Activation memory: ~80% reduction |
| bfloat16 mixed precision | Weights + optimizer: 50% reduction |
| Continuous batching generation | Generates completions efficiently without vLLM |

**Effective VRAM breakdown at batch_size=2:**
- 4-bit Model weights: ~2.5 GB
- LoRA adapters: ~0.1 GB
- Optimizer states (8-bit): ~0.3 GB
- KV cache (2×1024 tokens): ~0.5 GB
- Activations (with grad ckpt): ~1.5 GB
- Overhead: ~2 GB
- **Total: ~6.9 GB** ✓ (fits in 8 GB)

### Run training

```powershell
# Activate the venv
C:\Users\Azooo\arabic-reasoning-rlvr-sota\.venv\Scripts\Activate.ps1

# Navigate to project
cd C:\Users\Azooo\arabic-reasoning-rlvr-sota

# Run with the local QLoRA config
python -m rlvr_sota train `
  --config configs/qwen_4b_qlora.yaml `
  --data data/example_math.jsonl

# With wandb logging (set up wandb first: wandb login)
python -m rlvr_sota train `
  --config configs/qwen_4b_qlora.yaml `
  --data data/example_math.jsonl `
  --wandb

# Quick test — just 10 steps to verify everything works
python -m rlvr_sota train `
  --config configs/qwen_4b_qlora.yaml `
  --data data/example_math.jsonl `
  --max-steps 10

# With a different model (must match architecture for target_modules!)
python -m rlvr_sota train `
  --config configs/qwen_4b_qlora.yaml `
  --data data/math.jsonl `
  --model "Qwen/Qwen2.5-7B-Instruct"
```

### What you'll see during training

```
Model:    Qwen/Qwen2.5-3B-Instruct
Data:     data/math.jsonl
Loss:     dr_grpo
Sampling: token
4-bit:    True
G:        8

{'loss': 0.052, 'grad_norm': 0.8,
 'rewards/correctness_reward_func/mean': 0.65,
 'rewards/format_reward_func/mean': 0.78,
 'rewards/language_reward_func/mean': 0.92,
 'rewards/answer_leak_penalty_func/mean': -0.05,
 'rewards/structural_leak_penalty_func/mean': -0.01,
 'reward': 0.72, 'reward_std': 0.18,
 'kl': 0.003, 'entropy': 4.2,
 'completions/mean_length': 156,
 'clip_ratio/region_mean': 0.02,
 'epoch': 0.15}
```

**Key metrics to watch**:
- `reward`: Should increase over time (the model is learning)
- `kl`: Should stay small (< 0.1). If it spikes, increase `beta`.
- `entropy`: Should not collapse to near 0. Healthy range: 2-8 nats.
- `completions/mean_length`: Early warning for length-hacking. If it spikes without reward improvement, the model is padding.
- `clip_ratio/region_mean`: Fraction of tokens clipped. Should be < 0.3. High values mean the model is changing too fast.

### Expected training time (RTX 2080 Super)

| Dataset Size | Time (estimated) |
|---|---|
| 100 samples | ~5-10 minutes |
| 1,000 samples | ~45-90 minutes |
| 4,000 samples | ~3-6 hours |
| 10,000 samples | ~8-15 hours |

The bottleneck is generation (8 completions × 2 prompts × 1024 tokens each). Continuous batching helps significantly.

## 8. Running on cloud GPU (A100/H100)

For real training runs (thousands of samples, multiple epochs), rent a cloud GPU.

### Setup

```bash
# On cloud instance (Ubuntu)

# Clone or copy the project
git clone ... arabic-reasoning-rlvr-sota
cd arabic-reasoning-rlvr-sota

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install trl peft bitsandbytes accelerate wandb pyyaml pytest
pip install -e /path/to/arabic-reasoning-rlvr

# Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should show: NVIDIA A100-SXM4-80GB or NVIDIA H100

# Login to wandb
wandb login
```

### Run

```bash
# Full-precision or LoRA on cloud with larger batch
python -m rlvr_sota train \
  --config configs/cloud_a100.yaml \
  --data data/math.jsonl \
  --wandb

# With vLLM for 10-20x faster generation (Linux only!)
# Install: pip install trl[vllm]
# Then set in YAML: use_vllm: true, use_transformers_continuous_batching: false
```

### Cloud config differences from local

| Parameter | Local (8GB) | Cloud (80GB) |
|---|---|---|
| `load_in_4bit` | `true` | `false` |
| `per_device_train_batch_size` | 2 | 16 |
| `gradient_accumulation_steps` | 8 | 2 |
| `num_generations` | 8 | 16 |
| `max_completion_length` | 1024 | 2048 |
| `optim` | `paged_adamw_8bit` | `adamw_torch` |
| `num_train_epochs` | 1 | 3 |
| `use_wandb` | `false` | `true` |
| `lora_r` | 64 | 128 |

### Estimated cost (2026 rates)

| GPU | $/hr | 4K samples × 3 epochs | 10K samples × 3 epochs |
|---|---|---|---|
| A100 80GB | $1.39 | ~2-4 hours ($3-$6) | ~5-10 hours ($7-$14) |
| H100 80GB | $2.69 | ~1-2 hours ($3-$5) | ~3-6 hours ($8-$16) |

At 4B parameters with LoRA, a single GPU is sufficient (no multi-node needed).

## 9. The reward system — how the model is scored

The model generates a completion like:

```
<think>أولا أجمع العددين معا ثم أحسب الناتج بدقة عالية جدا</think><answer>4</answer>
```

This is scored by **5 independent reward functions**, each returning a value. TRL sums them with weights.

### Function 1: `correctness_reward_func` (weight: 0.6)

```python
# For math: extracts number from <answer>, compares to ground truth
# "16.0" matches 16 (numeric comparison with tolerance 1e-6)
# "16" matches 16.0 (type coercion)
# Returns: 1.0 if correct, 0.0 if wrong

# For logic: extracts JSON from <answer>, compares dict/list exactly
# {"A":1,"B":2,"C":3} matches {"A":1,"B":2,"C":3} → 1.0
# {"A":1,"B":2,"C":9} vs {"A":1,"B":2,"C":3} → 0.0 (NO partial credit!)
```

**No partial credit for logic**: If 2 out of 3 entity assignments are correct but 1 is wrong, the score is 0.0. This prevents the model from using a "guess two, ignore the third" strategy.

### Function 2: `format_reward_func` (weight: 0.2)

```python
# Checks that <think>...</think> and <answer>...</answer> tags exist
# Empty <think></think> → 0.0 (defends against empty-reasoning hack)
# Gibberish inside <think> (unique token ratio < 0.4) → 0.0
# Short reasoning (< 10 tokens) → -0.3 penalty
# Low unique token ratio (< 0.6) → -0.2 penalty
# Full, varied reasoning → 1.0
```

### Function 3: `language_reward_func` (weight: 0.2)

```python
# Classifies tokens in <think> block as Arabic or non-Arabic
# Excludes numbers and math operators from counting
# arabic_ratio = Arabic_word_tokens / total_word_tokens
# English reasoning → 0.0-0.1
# Arabic reasoning → 0.8-1.0
# Empty <think> → 0.0 (penalized)
# Pure math (just numbers/operators) → 1.0 (neutral)
```

### Function 4: `answer_leak_penalty_func` (weight: 0.5, NEGATIVE)

```python
# Checks if the <think> block contains direct-answer phrases:
#   "the answer is definitely", "الإجابة النهائية هي", "الإجابة هي"
# If found → returns -1.0 (weighted: -0.5)
# If clean → returns 0.0
# Empty <think> → returns 0.0 (no text to check)
```

This catches the hack where the model puts the answer directly in the `<think>` block instead of deriving it step-by-step.

### Function 5: `structural_leak_penalty_func` (weight: 0.3, NEGATIVE)

```python
# Checks for reasoning content OUTSIDE the <think>...</answer> structure
# Text before <think> or after </answer> > 5 words → penalty
# Returns -1.0 if triggered, 0.0 otherwise
```

This catches the hack where the model writes reasoning outside tags to dodge format reward.

### Full reward trace for a good completion

```
Completion: "<think>أولا أجمع العددين معا ثم أحسب الناتج بدقة عالية جدا وخطوة بخطوة للتأكد</think><answer>4</answer>"
Ground truth: 4, Domain: math

correctness:   1.0   × 0.6 = +0.60
format:        1.0   × 0.2 = +0.20  (17 varied Arabic tokens, proper tags)
language:      1.0   × 0.2 = +0.20  (all Arabic words)
answer_leak:   0.0   × 0.5 = +0.00  (no "الإجابة هي" phrase)
structural:    0.0   × 0.3 = +0.00  (no text outside tags)
                               ──────
Total reward:                      1.00  ✓
```

### Full reward trace for a hack attempt

```
Completion: "<think></think><answer>4</answer>"
Ground truth: 99, Domain: math

correctness:   0.0   × 0.6 = +0.00  (4 ≠ 99)
format:        0.0   × 0.2 = +0.00  (empty think)
language:      0.0   × 0.2 = +0.00  (empty think, penalized)
answer_leak:   0.0   × 0.5 = +0.00  (empty think, nothing to check)
structural:    0.0   × 0.3 = +0.00  (no text outside tags)
                               ──────
Total reward:                      0.00  ✓ (hack caught)
```

## 10. GRPO vs GSPO vs GSPO-token

All three use the same group-relative advantage formula:
```
A_i = (R_i - mean(R)) / (std(R) + 1e-4)
```

The difference is in how the policy ratio (π_θ / π_old) is computed and clipped.

### GRPO (`importance_sampling_level: "token"`)

```
For each token t in completion i:
    ratio_t = π_θ(token_t | context) / π_old(token_t | context)
    loss_t = -min(ratio_t × A_i, clip(ratio_t, 1-ε, 1+ε) × A_i)
    Loss = mean over all tokens
```

**Use when**: Each token's choice matters (fine-grained reasoning).

### GSPO (`importance_sampling_level: "sequence"`)

```
For the whole completion:
    seq_ratio = ∏ ratio_t  (product of all per-token ratios)
    loss_i = -min(seq_ratio × A_i, clip(seq_ratio, 1-ε, 1+ε) × A_i)
    Loss = mean over completions
```

**Use when**: The entire completion is one unit (logic puzzles, full derivations where partial answers are meaningless).

### GSPO-token (`importance_sampling_level: "sequence_token"`)

```
Sequence-level ratio with token-level clipping.
Newest variant from the GSPO paper (July 2025).
```

**Use when**: You want the stability of sequence-level ratios but the precision of token-level clipping.

### Switching between them

Just change one line in your YAML:
```yaml
importance_sampling_level: "token"   # GRPO
importance_sampling_level: "sequence" # GSPO
importance_sampling_level: "sequence_token" # GSPO-token
```

No code changes needed. The pipeline imports the correct trainer class automatically.

### Which one for Arabic reasoning?

- **Math problems** → GRPO (`"token"`). Each reasoning step matters.
- **Logic puzzles** → GSPO (`"sequence"`). The full assignment must be correct.
- **Mixed dataset** → GRPO is the safe default.

## 11. Monitoring and debugging

### TRL's built-in metrics (logged every `logging_steps` steps)

| Metric | Meaning | Healthy Range |
|---|---|---|
| `reward` | Weighted sum of all 5 reward functions | Should increase over training |
| `reward_std` | Standard deviation of rewards in batch | 0.1-0.3, not 0 (no diversity = reward hacking) |
| `kl` | KL divergence from reference model | < 0.1, spike means policy drifting too fast |
| `entropy` | Mean per-token entropy of completions | 2-8 nats, dropping to < 1 = entropy collapse |
| `clip_ratio/region_mean` | Fraction of tokens clipped | < 0.3, high means model changing too aggressively |
| `completions/mean_length` | Average completion length | Should be stable. Spike = length-hacking warning |
| `completions/clipped_ratio` | Fraction of truncated completions | Should be near 0 |
| `frac_reward_zero_std` | Fraction of prompts where all G completions got same reward | < 0.5, high means no diversity (model collapsed) |

### Per-reward-component metrics

TRL logs each reward function separately:
```
rewards/correctness_reward_func/mean      → correctness rate
rewards/format_reward_func/mean           → format compliance
rewards/language_reward_func/mean         → Arabic language ratio
rewards/answer_leak_penalty_func/mean     → leak detection rate (negative)
rewards/structural_leak_penalty_func/mean → structural leak rate (negative)
```

**Watch for**:
- `correctness` increases but `format` drops → model is solving correctly but losing structure. Adjust reward weights.
- `answer_leak_penalty_func/mean` becomes strongly negative → model is leaking answers. Increase `w_answer_leak` (third element of `reward_weights`).
- `language_reward_func/mean` drops → model is switching to English. Check your system prompt and add more Arabic examples.

### Wandb dashboard

When `use_wandb: true`:
1. Run `wandb login` once to authenticate
2. All metrics appear in real-time at wandb.ai
3. Sample completions are logged so you can read what the model is generating
4. Checkpoint artifacts are tracked

```powershell
# Enable wandb
python -m rlvr_sota train --config configs/qwen_4b_qlora.yaml --data data/math.jsonl --wandb
```

### Without wandb

Metrics are still printed to the console at every logging step. Checkpoints are saved to `output_dir`.

### Sample completions

With `log_completions: true` and `num_completions_to_print: 4`, the trainer prints actual model outputs:
```
--- Completions at step 50 ---
Prompt: ما هو ناتج 2 + 3؟
 Completion 1: <think>أجمع 2 و 3 معا لأحصل على الناتج 5</think><answer>5</answer>
 Completion 2: <think>العملية بسيطة أجمع العددين اثنان وثلاثة والناتج هو خمسة</think><answer>5</answer>
 Completion 3: <think>2+3=5 مباشرة</think><answer>5</answer>
 Completion 4: <think></think><answer>5</answer>  ← hack!
```

This lets you spot problems immediately: empty think blocks, English reasoning, answer leaks.

### Entropy collapse detection

Entropy collapse is when the model's output distribution narrows so much it produces near-deterministic (and often degenerate) completions. It's a leading cause of reward hacking.

**Early warning signs** (monitor in wallb or console):
- `entropy` drops below 2.0 nats
- `frac_reward_zero_std` rises above 0.5 (all completions for a prompt get the same reward)
- `completions/mean_length` plateaus or drops
- `clip_ratio/region_mean` spikes

**If you see collapse**:
1. Increase `temperature` (e.g., 0.9 → 1.2)
2. Increase `beta` (e.g., 0.04 → 0.1, stronger KL reference model pull)
3. Decrease `learning_rate`
4. Add more diverse data to your dataset
5. The v1 codebase (`rlvr.entropy_guard`) has Clip-Cov/KL-Cov implementations you can use as post-hoc analysis

## 12. The test suite

### Running tests

```powershell
cd C:\Users\Azooo\arabic-reasoning-rlvr-sota
pytest -v
```

You'll see ~39 tests pass across 5 test files:

```
tests/test_rewards.py      — 13 tests  (reward function correctness)
tests/test_data.py         —  5 tests  (JSONL loading + validation)
tests/test_config.py       — 10 tests  (YAML parsing, GRPOConfig building)
tests/test_trainer.py      —  6 tests  (full GRPO training with tiny model)
tests/test_cli.py          —  5 tests  (CLI argument parsing)
```

### What the integration tests verify

The `test_trainer.py` integration tests run **actual GRPO training** with a randomly-initialized tiny GPT2 model:

1. `test_trainer_constructs_with_tiny_model` — GRPOTrainer builds successfully
2. `test_trainer_runs_training_steps` — 2 training steps complete, loss is produced, global_step > 0
3. `test_trainer_reward_functions_called` — All 5 reward functions are invoked during training
4. `test_trainer_gspo_mode` — GSPO mode runs without crash (verify flag switching)
5. `test_trainer_dapo_loss` — DAPO loss type runs without crash
6. `test_trainer_saves_checkpoint` — Model checkpoint is saved to disk

These tests prove the entire pipeline — generate, reward, advantage, loss, backprop, checkpoint — works end-to-end before touching a real model.

### Running the v1 test suite (100 additional tests)

```powershell
cd C:\Users\Azooo\arabic-reasoning-rlvr
.\.venv\Scripts\Activate.ps1
pytest -v
```

This validates all the reward math, advantage formulas, and schema validation logic.

## 13. Troubleshooting

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

Solutions (try in order):
1. Reduce `per_device_train_batch_size` from 2 to 1
2. Reduce `max_completion_length` from 1024 to 512
3. Reduce `num_generations` from 8 to 4
4. Set `bf16: true` (if not already)
5. Set `gradient_checkpointing: true`
6. If still failing, reduce `lora_r` from 64 to 32

### Slow generation

```
# If each step takes > 2 minutes
```
Solutions:
1. Ensure `use_transformers_continuous_batching: true` (it's **much** faster)
2. Reduce `num_generations` (8 → 4)
3. Reduce `max_completion_length` (1024 → 512)
4. On cloud Linux, install vLLM (`pip install trl[vllm]`) and set `use_vllm: true`

### Model generates garbage / non-Arabic

1. Verify the system prompt is set correctly in your config
2. Check `rewards/language_reward_func/mean` — if it's low, the model isn't being penalized enough. Increase `reward_weights` third element.
3. The base model (Qwen 2.5 3B Instruct) has some Arabic capability but weak Arabic reasoning. This is expected to improve during training.
4. If the model never produces Arabic at all, check your data: is the prompt in Arabic? Does the system prompt specify Arabic?

### Reward hacking patterns (and fixes)

| Pattern | Symptom | Fix |
|---|---|---|
| Empty `<think></think>` | `format_reward` drops, `correctness` may be high | Already caught by format_reward (returns 0.0 for empty think) |
| English reasoning | `language_reward` drops, `reward` drops | Works as designed — model learns to use Arabic |
| Answer in `<think>` ("الإجابة هي X") | `answer_leak_penalty` triggers | Already caught by answer_leak penalty |
| Reasoning outside tags | `structural_leak_penalty` triggers | Already caught by structural_leak penalty |
| Repeated tokens in `<think>` | `format_reward` drops due to low unique token ratio | Already caught by format_reward (<0.4 unique ratio = 0.0) |
| Length hacking (padding) | `completions/mean_length` spikes without reward improvement | Enable `mask_truncated_completions: true`, increase KL `beta` |
| Copying prompt as answer | `correctness` stays 0 (unless prompt has answer) | Add explicit "do not repeat" to system prompt, or add regex-based penalty |

### Training is unstable / loss oscillates

1. Reduce `learning_rate` (1e-5 → 5e-6)
2. Increase `beta` (0.04 → 0.1, stronger KL penalty keeps model closer to reference)
3. Ensure `max_grad_norm` is set to 1.0
4. Check if `epsilon_high` is too permissive (lower from 0.28 to 0.22)

### Can't find the model

```
OSError: Can't load tokenizer for 'Qwen/Qwen2.5-3B-Instruct'
```

The model needs to be downloaded from HuggingFace on first use (~8GB). Either:
- Have internet and it downloads automatically (cached in `~/.cache/huggingface/`)
- Or download it first: `huggingface-cli download Qwen/Qwen2.5-3B-Instruct`
- Or use a local path: `model_name: "/path/to/local/model"`

### Random seed for reproducibility

Set `seed: 42` in your config. Note that GPU operations (especially generation) have inherent non-determinism. For fully reproducible runs, also set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
But this slows down training significantly.

## 14. Extending the pipeline

### Adding a new reward function

1. Create the function in `rlvr.reward_composer` (Codebase 1):
```python
def reward_novelty(completion: str) -> float:
    # Your logic here
    return score
```

2. Add a TRL wrapper in `rlvr_sota.rewards` (Codebase 2):
```python
def novelty_reward_func(prompts, completions, **kwargs):
    return [float(reward_novelty(extract_text(c))) for c in completions]
```

3. Add it to `ALL_REWARD_FUNCS` and update `DEFAULT_REWARD_WEIGHTS`:
```python
ALL_REWARD_FUNCS = [correctness, format, language, leak, structural, novelty]
DEFAULT_REWARD_WEIGHTS = [0.6, 0.2, 0.2, 0.5, 0.3, 0.1]
```

4. Update the YAML `reward_weights` to match.

### Adding a new loss type

If a future TRL version adds a new loss type (e.g., "gspo_v2"), just set it in YAML:
```yaml
loss_type: "gspo_v2"
```

If the loss type requires additional GRPOConfig params, add them to `SOTAConfig` and pass them in `build_grpo_config()`.

### Using the failure bank for curriculum learning

The `rlvr.failure_bank` module banks prompts where the model gets zero reward on ALL G completions. Replay them in a hard-stage curriculum:

```python
from rlvr.failure_bank import set_current_stage, retrieve_banked_failures

# Stage 1: Easy samples (curriculum_stage: "easy")
# After stage 1 completes...

# Stage 2: Hard samples (curriculum_stage: "hard")
# The failure bank replays previously-failed prompts automatically
set_current_stage("hard")
```

To use this with the SOTA pipeline, modify `trainer.py` to switch datasets between stages.

### Adding cold-start SFT warm-up

Before GRPO training, you can fine-tune the model on cold-start CoT data (where the model sees examples of correct `<think>...</think><answer>...</answer>` responses):

```python
from rlvr_sota.data import load_cold_start_sft_dataset
from trl import SFTTrainer

sft_data = load_cold_start_sft_dataset("cold_start.jsonl", system_prompt)

# Standard SFT training (not GRPO)
# ... standard transformers Trainer with SFT data

# Then proceed to GRPO training with the SFT'd model as the starting point
```

### Using the difficulty labeler

The `rlvr.difficulty_labeler` can assign empirical difficulty quartiles to your data:

```python
from rlvr.difficulty_labeler import compute_error_rate, assign_difficulty_quartiles

# Run the base model on each sample 20 times to compute error rate
error_rates = [compute_error_rate(base_model, sample, n_attempts=20) for sample in samples]

# Assign quartile labels (trivial/easy/medium/hard)
refined = assign_difficulty_quartiles(samples, error_rates)

# Use refined difficulty for curriculum learning
easy_samples = [s for s in refined if s["difficulty_tag"] in ("trivial", "easy")]
hard_samples = [s for s in refined if s["difficulty_tag"] in ("medium", "hard")]
```
