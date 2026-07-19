# Pilot samples (200 + 200)

Quality reference before full-scale **4,000 CoT + 4,000 RLVR**.
See [`../GENERATION_PLAN.md`](../GENERATION_PLAN.md) for production requirements and how to generate.

## Files
- `coldstart_200.jsonl` — SFT CoT traces (Pro teacher, soft-polish dial)
- `rlvr_200.jsonl` — RLVR prompts (empty response) with pass@8 band fields
- `rlvr_deferred.jsonl` — 0/8 passes (held out of the 200)
- `pass8_calibration.json` — full pass@8 results
- `STATS.json` — domain / steps / band mixes
- `READABLE_*.md` — human-readable previews

## Protocol
- **Generation:** `deepseek-v4-pro` + GEPA ArabicTeacher (materialize + soft polish)
- **Difficulty (RLVR):** Flash `pass@8` proxy + hybrid fallback when bands collapse
  - keep Flash `easy` / `medium` / `hard`
  - Flash `deferred` (0/8) → sampling `hard`
  - Flash `mastered` (8/8) → teacher `difficulty_tag` / grade / steps
- Target RLVR mix: ~30% easy / 50% medium / 18% hard (+ hard diagnostics)
- Domains: gsm8k, math, math_comp, logic (~30 / 25 / 25 / 20 at full scale)

## Important
Flash pass@8 is **provisional**. On this pilot Flash was too strong on math (mostly 8/8)
and failed all logic (0/8). Recalibrate on the real student checkpoint before production scale.
