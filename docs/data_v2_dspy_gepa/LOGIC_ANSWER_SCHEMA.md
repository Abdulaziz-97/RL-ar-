# Logic answer schema (frozen)

Corpus: `v1_frontier_regen`

## Reward-safe contract

| Field | Type | Rule |
|-------|------|------|
| `metadata.ground_truth_answer` | **flat `dict`** | Keys/values JSON-serializable strings (or stringifiable). Never stringify the whole GT into meta. |
| top-level `answer` | **string** | Canonical JSON: `json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` |
| cold `<answer>…</answer>` | **string** | Same canonical JSON as top-level `answer` |

## Matching

Training reward must use `answers_match_logic(model_ans, gt)` from `verify_budget_v3.py`:

- Accept model answer as dict **or** JSON string
- Flatten one level of nesting if needed
- Compare normalized `str` key/value maps for equality

Unit test: `python -m synth.test_logic_reward`

## Example

```json
{
  "answer": "{\"أحمد_مهنة\":\"طبيب\",\"سارة_مهنة\":\"مهندسة\"}",
  "metadata": {
    "ground_truth_answer": {
      "أحمد_مهنة": "طبيب",
      "سارة_مهنة": "مهندسة"
    }
  }
}
```

## Banned

- Pretty-printed / unsorted JSON as the scored `answer` string
- Nested dict as the only stored GT without a flat meta dict
- Putting the full logic JSON inside `<think>`
