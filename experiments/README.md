# Experiment grid (team_pack)

Ablation grid for Arabic Reasoning RLVR inside this pack.

**Start here:** [HOW_TO_RUN_EXPERIMENTS.md](HOW_TO_RUN_EXPERIMENTS.md)

```bash
cd team_pack
$env:PYTHONPATH = "src;."   # Windows
python -m experiments validate
python -m experiments list
```

- `grid.yaml` — 41 experiments / 10 phases  
- Default train module: `rlvr_pipeline`  
- Default config: `configs/qwen_4b_qlora.yaml`
