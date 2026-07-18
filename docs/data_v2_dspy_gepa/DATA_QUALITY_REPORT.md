# Data Quality Report — `v2_dspy_gepa`

**Verdict: PASS — ready to train**

## Must-fix confirmation

| Item | Value |
|------|-------|
| Cold train / eval | 640 / 160 |
| Cold train ∩ eval IDs | **0** |
| Gate train_eval_id_overlap | {"ok": true, "rlvr_overlap": 0, "cold_overlap": 0, "cold_train_n": 640, "cold_eval_n": 160, "rlvr_train_n": 640, "rlvr_eval_n": 160} |

See `SPLIT_FIX_NOTE.md`.

## Ship gate checks

| Check | Status | Detail |
|-------|--------|--------|
| `no_trailing_salt` | **PASS** | {"detail": 0} |
| `no_marji_junk` | **PASS** | {"detail": 0} |
| `no_mutabaa_junk` | **PASS** | {"detail": 0} |
| `no_gt_in_prompt` | **PASS** | {"rate": 1.0, "fail_n": 0} |
| `cold_xml_format` | **PASS** | {"rate": 1.0} |
| `rlvr_answer_filled` | **PASS** | {"rate": 1.0} |
| `no_final_gt_leak_think` | **PASS** | {"rate": 1.0} |
| `flash_agree_honest` | **PASS** | {"soft_pass_n": 0, "verify_method_lie_n": 0, "flash_agree_true_n": 800} |
| `flash_agree_coverage` | **PASS** | {"rate": 1.0, "true_n": 800, "mode": "required"} |
| `no_mmlu_in_sft` | **PASS** | {"detail": 0} |
| `logic_schema` | **PASS** | {"rate": 1.0} |
| `think_quality` | **PASS** | {"rate": 1.0} |
| `uniqueness` | **PASS** | {"near_dup_pairs_rlvr": 0, "near_dup_pairs_cold": 0} |
| `difficulty_labels` | **PASS** | {"rate": 1.0} |
| `sizes` | **PASS** | {"rlvr": 800, "cold": 800} |
| `unique_ids` | **PASS** | {"rlvr_duplicate_id_rows": 0, "cold_duplicate_id_rows": 0, "rlvr_unique": 800, "cold_unique": 800} |
| `train_eval_id_overlap` | **PASS** | {"rlvr_overlap": 0, "cold_overlap": 0, "cold_train_n": 640, "cold_eval_n": 160, "rlvr_train_n": 640, "rlvr_eval_n": 160} |
| `hard_share` | **PASS** | {"share": 0.4} |
| `domain_mix` | **PASS** | {"mix": {"math_comp": 180, "gsm8k": 280, "math": 180, "logic": 160}} |
| `cold_answer_matches_gt` | **PASS** | {"rate": 1.0} |

## Policy notes

1. **GT in prompt:** bare GT len>=2 as whole token banned; 1-digit and substring-in-larger-number (17 in 170) are intentional non-rejects.
2. **Math CoT `= N` last line:** allowed; classic closers only banned.
3. **مرجعًا noun:** legitimate word-problem noun; not stripped.
