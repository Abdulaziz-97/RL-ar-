# Data Explanation Report — Arabic Reasoning 2_dspy_gepa

## What this dataset is

Train-safe Arabic reasoning corpus for:

1. **SFT cold-start** — Formal Arabic chain-of-thought in <think> + canonical <answer>
2. **RLVR / GRPO** — prompt + programmatic ground truth (empty teacher response)

Corpus version stamp: **2_dspy_gepa**.

## Domains (800 + 800)

| Domain | n | Role |
|--------|---|------|
| gsm8k | 280 | Arabic word problems (arithmetic) |
| math | 180 | Short formal math |
| math_comp | 180 | Competition-style math |
| logic | 160 | Assignment / puzzle logic (flat JSON answers) |

## File map (data/)

| File | Use |
|------|-----|
| rabic_reasoning_coldstart_train.jsonl | **Primary SFT train** |
| rabic_reasoning_coldstart_eval.jsonl | SFT held-out |
| rabic_reasoning_rlvr_train.jsonl | **Primary RLVR/GRPO train** |
| rabic_reasoning_rlvr_eval.jsonl | RLVR held-out |
| rabic_reasoning_dataset_train.jsonl | Combined train (RLVR+cold) |
| rabic_reasoning_dataset_eval.jsonl | Combined eval |
| rabic_reasoning_coldstart.jsonl | Full cold (800) |
| rabic_reasoning_rlvr.jsonl | Full RLVR (800) |
| rabic_reasoning_dataset.jsonl | Full combined (1600) |
| human_review_sample_20.jsonl | Stratified cold sample for human QA |

## Row contracts

### RLVR row
- prompt: Arabic problem (no GT leak, no bureaucracy salt)
- 
esponse: empty
- nswer: canonical final (digits or compact JSON)
- metadata.ground_truth_answer: programmatic GT (dict for logic)
- metadata.verify_method: programmatic

### Cold row
- prompt: same style as RLVR
- 
esponse: <think>…</think><answer>…</answer>
- nswer: equals programmatic GT
- <think>: Formal Arabic reasoning; **must not restate final GT**
- metadata.flash_agree: **true** (required)
- metadata.cot_teacher: DSPy GEPA Pro teacher id

## How cold CoTs were produced

1. Compile DSPy ArabicTeacher with GEPA (uto=medium) → rabic_teacher_gepa_v2.json
2. Multi-stage module: plan → solve (split think/answer) → refine → strip leak → complete truncated → compact
3. Accept only if metric + Flash independent agree
4. Offline refilter (leak / boilerplate / incomplete) → **0 drops** on final 800
5. Ship gate green → production overwrite

## How to train

1. SFT on rabic_reasoning_coldstart_train.jsonl (format: think/answer XML)
2. Then RLVR/GRPO on rabic_reasoning_rlvr_train.jsonl with programmatic reward vs metadata.ground_truth_answer
3. Logic reward: see LOGIC_ANSWER_SCHEMA.md / nswers_match_logic
4. Re-SFT from scratch if prior runs used rejected packs

## What was excluded (do not use)

- Quarantined old Pro cold
- Interim STATUS folders / partial cleans
- Soft-pass / fake Flash rows
- Any row failing ship gate checks

## Quality pointer

Full gate + stats: DATA_QUALITY_REPORT.md / .json.


## Split integrity

Cold/RLVR train and eval are ID-disjoint after the 2026-07-17 uniquify+resplit. See `SPLIT_FIX_NOTE.md`.
