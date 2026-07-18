# Split fix note (2026-07-17)

## Issue

Cold train/eval shared 7 IDs because refill reused `cold_gsm8k_refill_*` IDs for different prompts (24 duplicate IDs among 800 rows → only 776 unique). Stratified split then placed the same ID in both train and eval.

## Fix

1. Uniquified colliding IDs (`…__uNNNN`) — **kept all 800 rows** (different prompts).
2. Re-ran stratified ~80/20 split with hard ID exclusivity (`train ∩ eval = ∅`).
3. Added gate checks: `unique_ids`, `train_eval_id_overlap`.
4. Re-shipped production + this pack.

## Confirmation

- Cold train = 640, eval = 160
- `train ∩ eval` ID overlap = **0**
- Gate `train_eval_id_overlap.cold_overlap` = 0
- Gate `pass` = true

## Optional marji

Remaining 4× `مرجعًا` are legitimate countable nouns ("reference book" in word problems), **not** bureaucracy stamps — left as-is.

## Gate policy notes (for AI team)

- **`no_gt_in_prompt`:** Rejects bare GT with length ≥ 2 as a whole token. Does **not** reject 1-digit GT or substring-inside-larger-number (e.g. 17 inside 170). Intentional.
- **Math CoT last `= N` line:** Allowed. Only classic closers (`الجواب النهائي` / `إذن…هو N`) are banned.
