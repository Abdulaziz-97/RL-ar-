# How to run the experiment grid (team_pack)

This folder is a self-contained copy of the ablation grid for the **team pack** Qwen RLVR pipeline (`rlvr_pipeline`).

| File | Role |
|------|------|
| `grid.yaml` | 41 experiments across 10 phases |
| `__main__.py` | CLI entry: `python -m experiments …` |
| `runner.py` / `builder.py` / `manifest.py` / `status.py` | Load grid, build CLI, run, resume |
| `tests/` | Unit tests for the grid runner |

## Setup

From the **team_pack** root (not the monorepo root):

```bash
cd team_pack

# Windows PowerShell
$env:PYTHONPATH = "src;."

# Linux / macOS
# export PYTHONPATH="src:."

# One-time: pack install (if not already)
pip install -e .
```

`PYTHONPATH` must include both `src` (training code) and `.` (so `experiments` imports resolve).

## Quick commands

```bash
# Validate grid.yaml
python -m experiments validate

# List all experiments
python -m experiments list
python -m experiments list --phase 3 -v

# Dry-run (prints argv only; writes dry_run.json under the output dir)
python -m experiments run --dry-run --only B1 --sft-checkpoint ./runs/sft

# SFT stage from the grid (C2)
python -m experiments sft --config configs/qwen_4b_qlora.yaml

# Run one RL experiment
python -m experiments run `
  --config configs/qwen_4b_qlora.yaml `
  --sft-checkpoint ./runs/sft `
  --only C1 `
  --max-steps 300

# Run one whole phase
python -m experiments run --sft-checkpoint ./runs/sft --phase 4 --max-steps 300

# Resume (skip experiments that already have EXPERIMENT_DONE.json)
python -m experiments run --sft-checkpoint ./runs/sft --resume
```

Outputs land in `./runs/experiments/<EXP_ID>/` (`run.log`, checkpoints, `EXPERIMENT_DONE.json`). A phase summary is written to `./runs/experiments/SUMMARY.json`.

## Recommended workflow

1. **SFT once** (shared checkpoint for most experiments):
   ```bash
   python -m experiments sft --config configs/qwen_4b_qlora.yaml
   # or: python -m rlvr_pipeline sft --config configs/qwen_4b_qlora.yaml --output ./runs/sft
   ```
2. **Phase 1 baselines** (`B1`, `B2`) — no SFT attached even if you pass `--sft-checkpoint`.
3. **Phase 2+** — pass `--sft-checkpoint ./runs/sft` (or your SFT output).
4. After each phase, pick a winner and bake those knobs into `configs/qwen_4b_qlora.yaml` (or pass a custom `--config`) before the next phase.
5. **Phase 10 HP sweep** only overrides a few HPs; it assumes the base YAML already holds the best stack.

## Phase map

| Phase | Ids | Question |
|-------|-----|----------|
| 1 baselines | B1–B2 | No-SFT lower bounds |
| 2 coldstart | C1 | SFT → Dr. GRPO |
| 3 is_level | G1–G3 | Token vs sequence IS |
| 4 reward_scale | S1–S3 | Group / batch / off |
| 5 loss | D1–D5 | Loss variants |
| 6 curriculum | L1–L4 | Schedules |
| 7 failure_mining | F1–F3 | Zero-variance strategies |
| 8 crps | P1–P2 | CRPS on/off |
| 9 full_stack | A1–A6 | Ablations of full stack |
| 10 hp_sweep | H* | Sweep on best base config |

## Team-pack caveats (read before long runs)

- Default train module is **`rlvr_pipeline`** (not the old `rlvr_sota` name).
- Cold-start data path in this grid is `data/arabic_reasoning_coldstart_train.jsonl`.
- **`F2` (`zero_variance_strategy: replay_buffer`)** will fail: replay is not implemented safely in this pack. Skip it (`--only F1` / `F3`) or edit that cell in `grid.yaml`.
- **`sequence_token` + CRPS/curriculum/direct_scoring** (e.g. some `A3`-style stacks) is rejected by the trainer. Prefer `G3` alone (discard + no CRPS + no curriculum) or drop `sequence_token` for full-stack runs.
- On RTX 20-series, SFT/GRPO apply an automatic AMP fix (LoRA in float32). That is expected; look for `QLoRA AMP fix` in the log.

## Tests

```bash
cd team_pack
$env:PYTHONPATH = "src;."
pytest experiments/tests -q
```
