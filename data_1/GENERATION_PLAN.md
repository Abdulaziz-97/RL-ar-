# Full generation plan — 4,000 CoT + 4,000 RLVR

This document is the production brief for scaling this pack beyond the dial-20
replay and the 200+200 pilot under `samples/`.

**Deliverables**

| Corpus | Count | Role |
|--------|------:|------|
| Cold-start / CoT (SFT) | **4,000** | Full Formal Arabic reasoning in `<think>` + canonical `<answer>` |
| RLVR | **4,000** | Prompt + verifiable `answer_spec` only (`response` empty) |

Do not ship a large dump that fails isolation or difficulty gates. Quality-controlled
tranches beat an irreversible one-shot generation.

---

## 1. Training requirements (must meet)

These requirements define what “good” means for the next corpus.

### 1.1 Behavior target

- Student must write step-by-step reasoning in **Modern Standard Arabic (فصحى)** inside `<think>…</think>`.
- Final result only inside `<answer>…</answer>`.
- Western digits `0–9` only (no Eastern digits).
- No empty / pre-closed think blocks; no English CoT; no dialect.

### 1.2 Two corpora, different jobs

| | Cold-start (CoT) | RLVR |
|--|------------------|------|
| Purpose | Format + reasoning priming (SFT) | Verifiable RL (GRPO-style) |
| `response` | Required, teacher CoT | Must be empty |
| Ground truth | In `<answer>` and metadata | In `answer_spec` only |
| Difficulty | Teacher metadata OK | **Empirical** via student `pass@8` |

### 1.3 Domains (this pack’s labels)

Map to the recommended skill mix (~30 / 25 / 25 / 20):

| Pack `domain` | Share of accepted rows | Intent |
|---------------|------------------------|--------|
| `gsm8k` | ~30% | Word arithmetic |
| `math` | ~25% | School math |
| `math_comp` | ~25% | Competition-style math |
| `logic` | ~20% | Constraint / assignment / ordering puzzles |

No single template family should dominate without review. Logic must vary the
constraint structure, not only names and surface stories.

### 1.4 Answer types (v1)

Supported only: `integer`, `rational`, `decimal_exact`, `decimal_approx`, `logic_json`.

**Rejected:** symbolic / sympy / free-form expression answers.

Logic GTs must be **flat** JSON objects (no nested dict/list values).

### 1.5 Isolation (hard gates)

- **Zero** prompt / family overlap between cold-start and RLVR.
- **Zero** train ↔ eval family overlap within each track.
- Assign `family_id` / `partition` **before** paraphrase, variants, or traces.
- Decontaminate against public benchmarks and against the other track.
- Near-duplicate prompts (high Jaccard) must be dropped.

### 1.6 RLVR difficulty — empirical `pass@8`

Teacher tags (`difficulty_tag`, `grade_level`, `num_steps`) are metadata only.

For every RLVR candidate, run the **current student** (or an agreed proxy) with:

- same system prompt, tokenizer / chat template, temperature, top-p/top-k,
- same max completion length, fixed seed list, checkpoint hash recorded,
- `N = 8` generations; `pass@8` = fraction of verifier passes (not unbiased pass@k).

**Bands**

| Band | Passes / 8 | Use |
|------|------------|-----|
| mastered | 8/8 | Downweight / exclude from RL mix (no learning signal) |
| easy | 6–7/8 | Keep |
| medium | 3–5/8 | Keep (primary frontier) |
| hard | 1–2/8 | Keep |
| deferred | 0/8 | Hold for later curriculum (no positive signal yet) |

**Target RLVR sampling mix (of the shipped 4,000)**

- **30%** easy
- **50%** medium
- **18%** hard
- **~2%** very-hard diagnostics (from hard / near-deferred, clearly labeled)

Recalibrate after each accepted student checkpoint, or when aggregate pass rate
moves by more than ~5 points.

> **Proxy note:** DeepSeek Flash can be used as a *temporary* proxy for banding
> while the real student is unavailable. Flash collapsed on the 200 pilot
> (mostly 8/8 on math, 0/8 on logic). Production scale must recalibrate on the
> real GRPO student. Until then, document `difficulty_source` and any hybrid fallback.

### 1.7 Cold-start CoT quality

Teacher pipeline in this pack (required for production CoT):

`plan → solve → refine → strip leak → complete → materialize → soft polish → compact`

Ship-ready think must:

- match ground truth under the executable verifier;
- show **explicit** intermediates (`a × b = c`, not vague “نحصل على الناتج”);
- finish every bare `a op b` as `a op b = c`;
- include an equation whose RHS is the final numeric answer (numeric domains);
- avoid rhetorical closers (`إذن/لذلك … هو N` without an equation);
- avoid boilerplate (`نراجع الاتساق…`, `الخطوة التوضيحية…`);
- stay Formal Arabic with adequate length / uniqueness;
- fail closed on empty, incomplete, or vague think (drop the row).

Soft polish is best-effort; do not wipe rows that only have minor bare-ops if
materialize already produced usable equations — but do not ship incomplete /
vague traces.

### 1.8 Length and format

- Prefer concise, complete CoT over padded filler.
- Align SFT trace length with the RL completion budget used in training
  (typical band ~640–768 tokens for completions; avoid systematically
  teaching traces that always hit the ceiling).
- Format grammar is mandatory: `<think>…</think>` then `<answer>…</answer>`.

### 1.9 Human review

- Stratified sample for review every release (domain × difficulty).
- Double-annotate at least 100 accepted rows per release (or 30% if the release
  is smaller than ~334 rows).
- Rubric: correct math, real CoT (not template scaffolding), MSA, no leak,
  answer matches GT.

---

## 2. Generation plan (how to reach 4,000 + 4,000)

### 2.1 Over-generate, then select

Keep rates are below 100%. Plan headroom:

| Track | Target kept | Suggested families to generate | Notes |
|-------|------------:|-------------------------------:|-------|
| Cold-start CoT | 4,000 | **~6,000–7,000** | Pilot keep ~65% with Pro + materialize + soft polish |
| RLVR candidates | — | **~8,000–10,000** | Need surplus after gates + `pass@8` band fill |
| RLVR shipped | 4,000 | from candidates | After band mix + decontam |

Generate in **tranches** (e.g. 500–1,000 families), run gates, merge, then continue.
Do not wait for one multi-day job with no checkpoints.

### 2.2 Pipeline stages (this pack)

```
family_partition → latent_spec → deterministic_solve → arabic_render
    → multi_trace (teacher CoT for SFT; empty for RLVR partition)
    → step_verify → concision → gates → select_quarantine → release_prep
```

Roles are wired by `backends/dspy_backend.py`:

- **live:** `vendor/synth/programmatic.py` problems + GEPA `ArabicTeacher`
- **replay:** frozen stages under `assets/replay_stages/` (dial-20 only)

### 2.3 Recommended models

| Step | Model | Why |
|------|-------|-----|
| Problem + teacher CoT generation | `deepseek-v4-pro` | Best CoT quality in fair compares |
| Temporary pass@8 proxy (optional) | `deepseek-v4-flash` | Cheap; **not** final student difficulty |
| Final pass@8 | Real training student checkpoint | Required before calling bands “production” |

GEPA weights: `assets/arabic_teacher_gepa_v2.json`.

### 2.4 Seeds and partitions

Use **disjoint seeds** for cold vs RLVR (example: cold `seed=1200…`, RLVR `seed=2200…`).

Run two separate jobs:

1. `partitions: { sft_train: 1.0 }` → cold-start CoT corpus  
2. `partitions: { rlvr_train: 1.0 }` → RLVR prompt corpus  

Then carve eval splits **by family**, not by reshuffling prompts across tracks.

### 2.5 Domain quotas (accepted 4,000 each)

Approximate counts per track:

| Domain | ~Share | ~Count / 4,000 |
|--------|--------|----------------|
| gsm8k | 30% | 1,200 |
| math | 25% | 1,000 |
| math_comp | 25% | 1,000 |
| logic | 20% | 800 |

### 2.6 RLVR band quotas (accepted 4,000)

| Band | Share | Count |
|------|------:|------:|
| easy | 30% | 1,200 |
| medium | 50% | 2,000 |
| hard | 18% | 720 |
| hard-diagnostic | ~2% | 80 |

Mastered → exclude or replace. Deferred → hold file, do not count toward the 4,000
unless later recalibration moves them into hard/diagnostic with a clear label.

---

## 3. How to generate with this pack

### 3.1 One-time setup

```bash
cd pro_polish_pipeline_pack
python -m venv .venv
# Windows:
.\.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` → `.env` and set `DEEPSEEK_API_KEY` for live runs.

### 3.2 Sanity: offline replay (no API)

```bash
python scripts/run_pipeline.py --mode replay
python scripts/verify_repro.py
```

Expect `"pass": true` vs `reference_output/sft_train.jsonl`.

### 3.3 Small live pilot (API)

```bash
python scripts/run_pipeline.py --mode live --model deepseek-v4-pro
```

Uses `configs/pilot_20.yaml` (20 families, seed 890). Output under `outputs/run/`.

### 3.4 Scale cold-start CoT (4,000 kept)

From the **parent repo** (full orchestrator + plugin), pattern used for the 200 pilot:

```bash
# Example shape — raise --cold-families until ≥4000 kept after gates
python scripts/build_data_team_samples_400.py \
  --cold-families 6500 \
  --target 4000 \
  --skip-rlvr --skip-pass8 \
  --budget-usd 200 \
  --work-root artifacts/full_scale_4000/cold_pro
```

Or call the pack backend live with a larger YAML:

```yaml
# configs/full_cold_4000.yaml  (create / adapt)
mode: cold_start
n_families: 6500          # over-generate; expect keep < 100%
domains: [gsm8k, math, math_comp, logic]
traces_per_problem: 1
seed: 1200
work_dir: outputs/full_cold
backend: external
external_module: backends/dspy_backend.py
max_alternate_methods: 0
partitions:
  sft_train: 1.0
```

```bash
python scripts/run_pipeline.py --mode live --model deepseek-v4-pro --config configs/full_cold_4000.yaml --work-dir outputs/full_cold
```

Export `release_corpora/sft_train.jsonl`, filter to 4,000 with domain balance, write
`coldstart_4000.jsonl`.

### 3.5 Scale RLVR (4,000 kept after bands)

1. Generate a large RLVR candidate pool (Pro, `rlvr_train` partition, empty responses).
2. Run `pass@8` on the student (or Flash proxy for interim banding).
3. Assign bands; fill 30/50/18/2; write deferred/mastered aside.
4. Enforce decontam vs cold-start and vs eval.

For separate jobs, set `decontam_reference_paths` in the later job's YAML to
the earlier release JSONL files. The gates load both prompts and responses and
fail if a configured reference is missing.

Pilot reference implementation: `scripts/build_data_team_samples_400.py` in the
parent repo (hybrid band fallback when Flash collapses). Extend `--target 4000`
and family counts; do **not** treat Flash bands as final.

### 3.6 Release checklist

Before calling a tranche “done”:

- [ ] Counts: 4,000 CoT + 4,000 RLVR (or tranche subset with clear IDs)
- [ ] Domain mix within tolerance of 30/25/25/20
- [ ] Cold ↔ RLVR prompt/family overlap = 0
- [ ] Train ↔ eval family overlap = 0
- [ ] All cold rows: format OK, answer matches GT, think quality gates pass
- [ ] All RLVR rows: non-empty prompt, valid `answer_spec`, empty `response`
- [ ] RLVR band mix ≈ 30/50/18/2 with `pass_at_8` + `difficulty_source` recorded
- [ ] Deferred / mastered written to sidecar files, not silently mixed in
- [ ] Manifest + `STATS.json` (domains, steps, bands, spend)
- [ ] Stratified human-review sample attached

### 3.7 Pilot artifacts already in this pack

| Path | What it shows |
|------|----------------|
| `samples/coldstart_200.jsonl` | CoT quality bar (Pro + materialize + soft polish) |
| `samples/rlvr_200.jsonl` | RLVR shape + banding fields |
| `samples/pass8_calibration.json` | Flash proxy pass@8 (collapsed — see STATS note) |
| `samples/STATS.json` | Mix / protocol for the pilot |
| `reference_output/` | Dial-20 golden SFT (replay SHA) |

Study these before full scale. Match **quality**, not only row count.

---

## 4. Schema sketch

### Cold-start row

```json
{
  "problem_id": "…",
  "domain": "gsm8k|math|math_comp|logic",
  "partition": "sft_train",
  "prompt": "…",
  "response": "<think>\n…\n</think>\n<answer>…</answer>",
  "answer_spec": { "type": "integer", "canonical": "42" },
  "metadata": {
    "num_steps": 3,
    "difficulty_tag": "medium",
    "ground_truth": 42
  }
}
```

### RLVR row

```json
{
  "problem_id": "…",
  "domain": "math",
  "partition": "rlvr_train",
  "prompt": "…",
  "response": "",
  "answer_spec": { "type": "integer", "canonical": "42" },
  "metadata": {
    "difficulty_band": "medium",
    "difficulty_source": "student_pass8|<checkpoint>",
    "pass_at_8": { "passes": 4, "n": 8, "pass_fraction": 0.5 }
  }
}
```

---

## 5. What not to do

- Do not reuse the old cold↔RLVR overlapping pack as production.
- Do not invent ground truth or soft-pass verifier failures.
- Do not ship vague CoT that “talks about” math without writing intermediates.
- Do not treat teacher `difficulty_tag` as empirical RL difficulty.
- Do not treat Flash `pass@8` as final once a student checkpoint exists.
- Do not put MMLU / private-eval labels into generation prompts.
- Do not commit API keys.

---

## 6. Pointers in this pack

| File | Role |
|------|------|
| `backends/dspy_backend.py` | Live vs replay wiring |
| `vendor/synth/dspy_teacher.py` | CoT teacher + gates |
| `vendor/synth/programmatic.py` | Deterministic problem factory |
| `src/rlvr_synth/orchestrator.py` | Stage DAG |
| `src/rlvr_synth/calibration/pass_at_n.py` | Band cutoffs |
| `src/rlvr_contracts/` | Format + answer_spec + verifiers |
| `README.md` | Setup + layout |
