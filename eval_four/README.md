# Core 4 Arabic Benchmark Evaluation (`eval_four`)

This folder contains a dedicated evaluation runner designed to evaluate models on the **Core 4 Arabic Benchmarks**:

1. **AraIFEval** (`humain-ai/AraIFEval`) - Instruction-Following & Constraint Satisfaction
2. **AraPro** (`humain-ai/AraPro`) - Professional Arabic Knowledge Accuracy
3. **AraTrust** (`humain-ai/AraTruthfulQA`) - Truthfulness & Common Sense Reasoning
4. **AraMath** (`humain-ai/AraMath`) - Arabic Mathematical Word Problem Solving

---

## How to Run on Vast.ai / GPU Instance

```bash
cd /workspace/RL-ar-
git pull origin Efficient-Arabic-Reasnoning-Pipeline

# Run evaluation on the Core 4 Benchmarks
python eval_four/run_eval_four.py \
  --model Qwen/Qwen3.5-4B \
  --adapter ./runs/grpo_v1 \
  --batch-size 16
```

### Quick Test Run (e.g. 100 samples per task):

```bash
python eval_four/run_eval_four.py \
  --model Qwen/Qwen3.5-4B \
  --adapter ./runs/grpo_v1 \
  --limit 100
```

---

## Target Baseline Scores (Base Qwen3.5-4B)

| Benchmark | Target Metric | Expected Base Score |
| :--- | :--- | :---: |
| **AraIFEval** | Constraint Satisfaction | **25.40%** |
| **AraPro** | Accuracy | **66.09%** |
| **AraTrust** | Accuracy | **84.87%** |
| **AraMath** | Accuracy | **68.93%** |
