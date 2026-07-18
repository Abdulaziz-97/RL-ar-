# TRAINING_HANDOFF — Arabic reasoning `v2_dspy_gepa`

## Status

**Gate pass=true** for corpus `v2_dspy_gepa` (DSPy GEPA teacher + Flash agree).

Built from:

- RLVR: programmatic offline (hard share ≥40%, no mutabaa/marji junk)
- Cold: Formal Arabic `<think>` via compiled `arabic_teacher_gepa_v2.json` (GEPA medium, fixed metric)
- Accept only if programmatic GT match + no final-GT leak + no boilerplate/incomplete + **Flash agree=True**
- Old Pro cold under `generated_ship_v1/quarantine_old_pro_cold_*` was **not** merged

## What to train on

| Artifact | Path |
|----------|------|
| **Train** | `arabic_reasoning_*_train.jsonl` |
| Eval | `arabic_reasoning_*_eval.jsonl` |
| Full | `arabic_reasoning_rlvr.jsonl`, `arabic_reasoning_coldstart.jsonl` |
| Pack | `deliverables_ship_v2_dspy/` (also mirrored under `deliverables_ship_v1/` by ship script) |
| Gate | `qa_report_ship_gate.json` |
| Teacher program | `generated_ship_v2_dspy/arabic_teacher_gepa_v2.json` |
| Logic schema | `LOGIC_ANSWER_SCHEMA.md` |
| Human sample | `human_review_sample_20.jsonl` |

## Ops (required)

1. Point configs at `*_train.jsonl` from this green ship.
2. **Re-SFT from scratch** if prior runs used rejected packs — do not resume collapsed GRPO / bad SFT.
3. Format probe after SFT; GRPO smoke: format>0, correctness>0, **KL ≪ 1**.
4. If format ≪ 0.5 → regenerate cold slice with `arabic_teacher_gepa_v2.json` only (never old Pro quarantine).

## Data contracts

- RLVR: empty `response`, filled canonical `answer`, programmatic GT.
- Cold: Formal Arabic `<think>`; `<answer>` = GT; **no final GT in think**; **Flash agree required**.
- Logic: meta GT = dict; `answer` = canonical JSON.
- No salt / `مرجع…برمز` / `رقم المتابعة` stamps.
- Digits in think: Western preferred; leak detector covers Eastern closers.

## Do not

- Train on `deliverables_dspy_v2_STATUS/` (interim-only)
- Merge quarantined old Pro cold
- Soft-pass Flash or invent GT / template CoT
- Call training done if `qa_report_ship_gate.json` shows `pass=false`
