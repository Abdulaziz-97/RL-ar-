# Human review rubric (20-row sample)

File: `human_review_sample_20.jsonl` (stratified cold rows)

Score each row 0/1 on:

1. **Prompt clean** — Formal MSA; no trailing salt `(…)`; answer not leaked as “الجواب: …”
2. **XML format** — `<think>…</think><answer>…</answer>` only; no `####`
3. **Real CoT** — Reasoning looks teacher-written; no `الخطوة التوضيحية` / filler step lists
4. **GT match** — Final `<answer>` equals programmatic GT (numeric/logic/math_comp)
5. **Think hygiene** — Final GT / full logic JSON absent from `<think>`; intermediates OK
6. **Think quality** — ≥~20 tokens, not repetitive, Arabic-dominant

Pass bar for ship confidence: ≥18/20 rows all-1 on items 1–6.

Notes column: free-text issues for regen slice if GRPO format ≪ 0.5 later.
