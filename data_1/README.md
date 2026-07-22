# Pro + Materialize + Soft-Polish Pipeline Pack

Portable folder to reproduce the fair dial run (20 families, seed 890, DeepSeek Pro)
and to scale to **4,000 CoT + 4,000 RLVR** with the same pipeline.

**Start here for full-scale work:** [`GENERATION_PLAN.md`](GENERATION_PLAN.md)
(requirements, domain/difficulty mix, how to generate, release checklist).

## Setup

```bash
cd pro_polish_pipeline_pack
python -m venv .venv
# Windows:
.\.venv\Scripts\pip install -e .
```

Copy `.env.example` to `.env` and set the API key for the provider you use
(at minimum `DEEPSEEK_API_KEY` for `--provider deepseek`).

## Reproduce our output (no API)

```bash
python scripts/run_pipeline.py --mode replay
python scripts/verify_repro.py
```

`verify_repro.py` must print `"pass": true` and matching SHA-256 vs `reference_output/sft_train.jsonl`.
After generation, run `rlvr-synth ship --work-dir outputs/run --release-id <id>`
to build an immutable manifest and execute the release gates.

## Live generation (multi-provider API)

```bash
# List providers
python scripts/run_pipeline.py --list-providers

# DeepSeek (default)
python scripts/run_pipeline.py --mode live --provider deepseek --model deepseek-v4-pro

# OpenRouter
python scripts/run_pipeline.py --mode live --provider openrouter --model deepseek/deepseek-chat-v3-0324

# DashScope / Qwen
python scripts/run_pipeline.py --mode live --provider dashscope --model qwen-plus

# OpenAI
python scripts/run_pipeline.py --mode live --provider openai --model gpt-4o

# Any OpenAI-compatible server
# .env: RLVR_API_BASE=https://...  RLVR_API_KEY=...
python scripts/run_pipeline.py --mode live --provider custom --model my-model
```

Env overrides: `RLVR_PROVIDER` and `RLVR_MODEL`. Registered providers use
their provider-specific key variable; `RLVR_API_BASE` and `RLVR_API_KEY` are
reserved for `--provider custom` to prevent stale generic credentials from
silently redirecting another provider.
For providers whose model pricing is not fixed in the registry (for example
OpenRouter or a custom endpoint), also set
`RLVR_PRICE_INPUT_PER_MILLION` and `RLVR_PRICE_OUTPUT_PER_MILLION`. Live runs
fail closed without those rates so the USD budget cannot silently undercount.

Live mode will not match reference hashes (LLM sampling). Teacher stages (GEPA
signatures + materialize/polish) stay the same; only the LM endpoint changes.

For production volume (4,000 + 4,000), follow **[`GENERATION_PLAN.md`](GENERATION_PLAN.md)** —
over-generate, gate, band with `pass@8`, then select.

## Layout

| Path | Role |
|------|------|
| `GENERATION_PLAN.md` | **Full-scale brief:** requirements + how to generate 4k CoT / 4k RLVR |
| `scripts/run_pipeline.py` | CLI: replay (offline) or live (API) generation |
| `scripts/verify_repro.py` | SHA-check replay output vs `reference_output/` |
| `backends/dspy_backend.py` | Glue: replay cache **or** live programmatic + teacher |
| `src/rlvr_synth/orchestrator.py` | Stage DAG (partition → render → traces → release) |
| `src/rlvr_contracts/` | `<think>/<answer>` parse, answer_spec, verifiers |
| `src/rlvr_synth/calibration/pass_at_n.py` | pass@8 band helper (`mastered`…`deferred`) |
| `vendor/synth/programmatic.py` | Deterministic Arabic problem generators + scrubbers |
| `vendor/synth/dspy_signatures.py` | DSPy Signature field constraints for the teacher |
| `vendor/synth/providers.py` | Multi-provider registry (DeepSeek, OpenAI, OpenRouter, …) |
| `vendor/formal_saudi_style.py` | MSA style prompt + dialect scorers (legacy name) |
| `vendor/answer_match.py` | Fast numeric/logic equality for teacher metric |
| `assets/arabic_teacher_gepa_v2.json` | GEPA teacher |
| `assets/replay_stages/` | Frozen stages from the dial run |
| `reference_output/` | Golden `sft_train.jsonl` (17 kept) |
| `samples/` | Pilot: 200 cold-start CoT + 200 RLVR |
| `configs/pilot_20.yaml` | seed=890, 20 families |

## Reference settings

- seed: `890`
- n_families: `20`
- kept SFT: `17`
- model (live): `deepseek-v4-pro`
- pipeline: plan → solve → refine → strip → complete → materialize → polish(soft) → compact
- reference SHA-256: `c41211fa96c5076877bd22fef456ff0424113139ec7182697c3ba5ae952d27bb`

## Pilot samples

See `samples/` for 200 cold-start + 200 RLVR (Pro gen, Flash pass@8 proxy).
Full-scale targets and requirements: [`GENERATION_PLAN.md`](GENERATION_PLAN.md).
