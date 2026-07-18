# Proposal: Production-Grade Arabic Reasoning Data for SFT → RLVR

**Audience:** Synthetic Data Team, Training Team, Evaluation Team  
**Student model:** Qwen3.5-2B, QLoRA  
**Target behavior:** Arabic step-by-step reasoning in `<think>`, canonical final result in `<answer>`  
**Status:** Generation specification; Phase 0 blocks even the 500-family pilot

---

## Executive decision

Do **not** scale the current `v2_dspy_gepa` recipe unchanged.

The current pack proved useful for finding pipeline bugs, but it is not a trustworthy production corpus:

- 586/800 prompts are shared between the full cold-start and RLVR pools.
- 379/640 cold-train prompts occur in RLVR-train.
- 91/160 RLVR-eval prompts occur in cold-train.
- All 1,600 rows have a programmatic `quality.score = 1.0`; this is not a measured quality distribution.
- The 20-row human-review sample has no completed review scores.
- The SFT length distribution conflicts with the RL reward.
- The existing policy forces unnatural reasoning that avoids naturally stating a derived result.
- Current evaluation cannot distinguish memorization from generalization.

The training pipeline also had two fatal non-data bugs in the tested configuration:

1. Qwen3.5's chat template inserted a pre-closed empty `<think></think>` into rollout prompts.
2. The SFT LoRA adapter silently failed to load into GRPO.

After fixing both, v11 improved format reward from an old peak of 0.29 to 0.57–0.82 and Arabic consistency to 0.90–0.98. This strongly indicates that earlier format comparisons were confounded by pipeline defects. However, correctness remained only 0.06–0.25, completions remained near the 512-token ceiling, and the run crashed at step 7 with a CUDA/CUBLAS failure.

The next corpus must therefore optimize for **verified correctness, learnable difficulty, concise reasoning, diversity, and clean evaluation**—not row count.

---

## 1. What happened in previous runs

### 1.1 Pipeline failures that invalidated v4–v10b

#### Pre-closed reasoning block

The tokenizer rendered this before model generation:

```text
<|im_start|>assistant
<think>

</think>

```

The model was told that reasoning had already ended, while the reward required a new non-empty `<think>` block inside the completion. Format reward was structurally difficult to earn.

The fixed probe changed from:

```text
<answer>42</answer>
```

to valid full reasoning, with 3/3 probe samples scoring format 1.0.

#### SFT adapter was not loaded

The SFT checkpoint used keys under `model.language_model.layers.*`; the GRPO model expected `model.layers.*`. The old loader also nested an already-PEFT-wrapped model. PEFT warned but did not fail, so GRPO effectively trained from the base model.

The strict loader now remaps and verifies all 372 LoRA tensors. Future training must hard-fail if any adapter tensor is missing.

### 1.2 Training failures that remain relevant

- Nonzero KL penalty caused KL values around 4–6; `beta=0` was required.
- Continuing from a low-entropy GRPO checkpoint accelerated collapse.
- Adaptive temperature reduced hard entropy collapse but did not fix correctness.
- Sixteen generations per prompt on an 8 GB RTX 2080 Super are slow and memory-fragile.
- v11 reached the correct format but still produced only 6–25% correct answers.
- v11 completion lengths were 462–512 tokens, with clipped ratios around 19–31%.
- v11 crashed at step 7 with `CUBLAS_STATUS_EXECUTION_FAILED` during backward.

### 1.3 What these results mean for data

The data must:

- produce a much stronger correctness prior before RL;
- teach concise, complete traces that reliably close both tags;
- contain enough problems at the student's learning frontier to create mixed reward groups;
- avoid train/eval contamination;
- use executable answer verification rather than teacher confidence;
- match the exact runtime answer and format contracts.

Data alone cannot compensate for broken prompt rendering, adapter loading, reward semantics, clipping, or unstable GPU settings. Those are Phase 0 blockers. The Data, Training, and Evaluation leads must sign the Phase 0 manifest before the pilot begins.

---

## 2. Audit of the current corpus

### 2.1 Current size and domains

Each stage contains 800 full rows, split 640 train / 160 eval:

- GSM-style arithmetic: 280
- general math: 180
- competition-style math: 180
- logic: 160

### 2.2 Cross-stage contamination

Exact normalized prompt overlap:

- cold-train ∩ RLVR-train: **379**
- cold-train ∩ RLVR-eval: **91**
- cold-eval ∩ RLVR-train: **88**
- cold-eval ∩ RLVR-eval: **28**
- full cold ∩ full RLVR: **586**

The ship gate only verified train/eval ID separation *within* each stage. It did not verify cross-stage prompt or family separation. The current RLVR eval is contaminated and must not be reported as a clean generalization benchmark.

### 2.3 Length mismatch

Current cold-start response lengths under the actual Qwen tokenizer:

- median: 212 tokens
- p90: 331
- p95: 363
- p99: 400
- maximum: 546
- 20 rows exceed 384 tokens
- 4 rows exceed 448 tokens
- 2 rows exceed 512 tokens

Current `<think>` word lengths:

- median: 100 words
- p95: 166
- maximum: 220
- 60.4% exceed the reward's soft limit of 80 words
- 6.6% reach or exceed the reward's hard limit of 160 words

The SFT stage teaches a behavior that RL then penalizes. This must be corrected in both the data and reward before regeneration.

### 2.4 Arabic quality

- Cold-start Arabic purity: minimum 0.8899, mean 0.9967.
- RLVR prompt Arabic purity: minimum 0.6863, mean 0.8949.
- Seven RLVR rows are below 0.75.
- Thirty-three RLVR rows are below 0.80.

A single character/script ratio does not detect translated prose, unnatural syntax, terminology inconsistency, or pedagogical quality. Native-speaker review is required.

### 2.5 Human validation is incomplete

The rubric defines a 20-row review, but the sample contains no reviewer scores, adjudication, or notes. Claims such as “99% clean” or “ready to train” are not statistically supported.

### 2.6 Metadata and QA-artifact defects

- `difficulty_tag` is null for all 280 GSM8K rows in both cold and RLVR, preventing complete curriculum stratification.
- The QA report stamps `v1_frontier_regen`, while the active rows and documentation identify the corpus as `v2_dspy_gepa`.
- `runs/coldstart_purity_audit.json` describes a stale 797-row pre-v2 corpus and must not be used to judge the active 800-row pack.
- RLVR `solution_steps` contains generated debris in some rows (for example, concatenated fragments such as `a_nاكتمل الحساب`). It is currently unused, but production metadata must not contain unvalidated pseudo-labels.
- The current ship gate can pass incomplete prose: `cold_math_comp_0127` contains a truncated sentence despite `flash_agree=true` and `quality.score=1.0`.

Production releases must regenerate every QA artifact from the immutable release manifest, require matching corpus/version hashes, and fail if stale reports are present.

### 2.7 Artificial reasoning caused by the leak policy

The current policy bans the final ground-truth value from `<think>`. This produces unnatural sentences such as “the calculation confirms the result without stating it explicitly.”

That is the wrong objective. A valid derivation may naturally compute and state the result before placing its canonical representation in `<answer>`. What must be banned is unsupported answer copying—not mathematical derivation.

---

## 3. Phase 0: freeze the training contract before generation

The data team must receive versioned implementations—not prose descriptions—of these contracts.

### 3.1 Exact response grammar

```text
<think>
[non-empty reasoning]
</think>
<answer>[canonical answer only]</answer>
```

Hard rules:

- UTF-8 response matches Python regex `\A<think>\n.+\n</think>\n<answer>.+</answer>\Z` in DOTALL mode after normalizing CRLF to LF;
- exactly one `<think>` block and one `<answer>` block;
- response begins with `<think>\n`;
- exactly one LF separates `</think>` and `<answer>`;
- no leading/trailing whitespace outside the two blocks;
- no pre-closed or empty think block;
- no Markdown fences;
- no prose before `<think>` or after `</answer>`;
- Arabic explanatory prose;
- equations, variables, standard symbols, Western digits, and JSON are exempt from Arabic-script scoring;
- `<answer>` contains only the canonical result.

The parser used for generation acceptance must be imported from the same versioned package used by training. Phase 0 must include positive and negative fixtures for whitespace, duplicated tags, missing closers, nested tags, empty bodies, prose outside tags, and truncation.

### 3.2 Answer types and canonicalization

Do not generate answer types that the runtime verifier cannot score.

Supported contracts must be explicit:

Every row must use a discriminated `answer_spec`:

```json
{
  "type": "integer|rational|decimal_exact|decimal_approx|symbolic|logic_json",
  "canonical": "string",
  "tolerance": null,
  "unit": null,
  "normalizer_version": "answer-v1",
  "verifier_artifact": "verifiers/<id>.json"
}
```

`tolerance` is required only for `decimal_approx`; `unit` is required only when units are scored. Negative zero normalizes to `0`; scientific notation and thousands separators are forbidden in canonical answers unless their normalizer explicitly supports them. JSON serialization is UTF-8, `sort_keys=true`, compact separators, and no NaN/Infinity.

#### Numeric

- integer: canonical decimal string, e.g. `-12`
- exact rational: reduced `p/q`, e.g. `3/4`
- finite decimal: normalized decimal string
- approximate decimal: only when the item declares a tolerance
- percentage/unit answer: only when the verifier explicitly normalizes it

Use exact arithmetic or symbolic normalization where possible. Float tolerance must never be the default for exact symbolic tasks.

#### Symbolic math

Either:

1. implement and version a SymPy-based equivalence verifier before generation, or
2. exclude symbolic-expression answers from the production pack.

Exact string equality is insufficient for equivalent expressions.

#### Logic

- `metadata.ground_truth_answer`: flat JSON object
- top-level `answer`: compact canonical JSON
- sorted keys
- no nested object unless the runtime verifier explicitly supports it
- assignment must be re-executed against the original constraints
- the puzzle must have a unique accepted solution, unless all valid solutions are represented by an explicit equivalence contract

The schema therefore stores both `answer_spec.canonical` (string used by the scorer) and `ground_truth_structured` (object used by the executable verifier).

### 3.3 Reasoning policy

Allow a derived final value to occur naturally in `<think>`.

Reject:

- an answer asserted before supporting work;
- copying from a visible reference answer;
- references to “ground truth,” “expected answer,” or verifier output;
- circular verification;
- repeated filler or restatement;
- a correct final answer reached through materially incorrect reasoning.

Do not reject a legitimate final arithmetic line such as `44 × 2 = 88` when it follows the derivation.

### 3.4 Length policy

Measure model tokens, not only whitespace-delimited words.

Recommended accepted-response targets:

- per-release easy p95 ≤ 220 completion tokens
- per-release medium p95 ≤ 360 completion tokens
- per-release hard p95 ≤ 520 completion tokens
- per-release overall p99 ≤ 512 tokens
- per-row hard rejection above 640 tokens unless expert-approved with a recorded reason

Reasoning should be as short as possible while preserving every necessary step. No quality metric should reward verbosity.

The runtime completion limit should be at least 640–768 tokens for validation runs, with fewer rollouts per prompt to fit memory. Production acceptance requires clipped ratio below 1%.

### 3.5 Runtime invariants

Before the data team generates a production batch, automated tests must prove:

- generated prompt ends at an open assistant turn, not a closed think block;
- SFT adapter tensor count and hashes match the checkpoint;
- LoRA gradients are nonzero;
- gold completions score correctness 1.0 and format 1.0;
- malformed and adversarial completions score 0;
- canonicalizers are identical offline and during training;
- checkpoint save/resume is functional;
- a multi-hour stress run completes without CUDA/CUBLAS failure.

---

## 4. Recommended production scale

Do not jump directly to hundreds of thousands of rows. Published small-model results show that a few thousand correctly pruned traces can produce gains and that additional synthetic volume can plateau; transfer to Arabic Qwen3.5-2B must be established experimentally.

Definitions used below:

- **family:** one latent problem structure, including all translations, paraphrases, numeric variants, and descendants;
- **problem:** one concrete prompt and verifier specification within a family;
- **candidate trace:** one separately sampled response for a problem;
- **accepted SFT trace:** one verified response selected for SFT;
- **RLVR prompt:** one prompt plus executable answer specification, without a stored training response.

All expansion counts are **cumulative**. The 500 pilot families count toward the 2,000-family cold-start target if they pass the same production gates.

### 4.1 Pilot

- 500 unique problem families, initially one concrete problem per family
- four separately sampled candidate traces per problem
- 2,000 raw traces
- retain approximately 700–900 verified SFT traces
- human review: at least 300 accepted rows plus a separate 100 rejected rows

No full-scale generation until the pilot passes every gate in Section 11.

### 4.2 Cold-start SFT production

- 2,000 unique accepted families cumulatively, initially one concrete accepted problem per family
- four candidate traces per problem
- retain one primary trace and at most one genuinely different valid method
- target 3,000–4,000 accepted SFT rows

After successful SFT and clean evaluation, expand to:

- 4,000–6,000 unique problems
- 7,000–10,000 accepted traces maximum

### 4.3 RLVR prompt pool

Initial:

- 3,000 unique prompts
- initially disjoint from SFT and always disjoint from evaluation by `family_id`
- no stored teacher response
- executable verifier for every row

Expansion after demonstrated gains:

- 6,000–10,000 unique RLVR prompts

### 4.4 Clean evaluation

- 1,000–2,000 private prompts
- frozen before production generation
- no prompts, paraphrases, templates, translations, constraint graphs, or numeric variants in SFT/RLVR training
- include native Arabic sources and new human-authored items
- keep labels inaccessible to generation models where operationally possible

### 4.5 Rejection-sampled second SFT

After an RL stage:

- select 1,000–2,000 informative prompts;
- retain 2,000–3,000 verified concise traces;
- prioritize corrected student failures, useful alternate methods, and difficult-but-learnable items;
- retrain SFT before the next RL stage.

This follows a demonstrated pattern: cold start → RL → rejection sampling → stronger SFT → RL.

---

## 5. Domain and difficulty design

### 5.1 Domain allocation

Recommended unique-problem mix:

- 30% `word_arithmetic`
- 25% `school_math`
- 25% `competition_math`
- 20% `logic_constraints`

These are mutually exclusive top-level domains. Algebra, geometry, number theory, probability, combinatorics, fractions, ratios, and related capabilities are multi-label `skills`, not competing domain labels.

No single sub-template should contribute more than 2% of accepted rows. No generator/template family should contribute more than 10% without explicit review.

Logic must vary the underlying constraint graph, not merely names and surface stories.

### 5.2 Arabic source mix

Target:

- at least 50% natively composed Arabic;
- at most 35% translated/adapted material;
- all translated items pass independent mathematical and native-Arabic review after translation.

`source_type` values are mutually exclusive (`native_ar`, `translated`, `synthetic_from_spec`, `derived`). Separately, at least 15% of all problems must set `arabic_context_native=true`; this attribute may coexist with any non-translated source type.

Translation is a source type, not proof of novelty.

### 5.3 Difficulty must be empirical

Teacher-assigned labels are metadata, not evidence.

For every RLVR candidate, sample the current student under the intended rollout settings and record `pass@N`.

For the first calibration, `N=8`, with the exact production system prompt, tokenizer/chat template, temperature, top-p/top-k, completion limit, seed list, and checkpoint hash recorded. `pass@8` is the raw fraction of eight executable-verifier passes; it is not an unbiased pass@k estimator. Recalibrate after each accepted model checkpoint or when aggregate pass rate shifts by more than five percentage points.

The most useful RL prompts produce mixed outcomes within a group:

- defer prompts with 0/N correct to a later curriculum: no current positive signal;
- reject or downweight prompts with N/N correct: no learning signal;
- prioritize prompts with approximately 1–(N−1)/N correct;
- keep a curriculum with enough easy/medium prompts to bootstrap correctness.

Initial RLVR mix:

- 30% easy/frontier-low
- 50% medium/frontier
- 18% hard/frontier-high
- 2% very hard diagnostics

As the student improves, recalculate difficulty and refresh the prompt pool.

Initial empirical bands under the fixed `N=8` protocol:

- easy: 6–7/8 correct;
- medium: 3–5/8;
- hard: 1–2/8;
- deferred: 0/8;
- mastered: 8/8.

### 5.4 Required skill coverage

Each domain manifest must enumerate actual skills:

- arithmetic operations, fractions, ratios, percentages, rates, unit conversion;
- equations, inequalities, systems, functions, sequences;
- geometry with text-sufficient diagrams/constraints only;
- number theory, counting, probability;
- multi-constraint assignment, ordering, satisfiability, set relations;
- verification, contradiction, and alternative-solution reasoning.

Image-dependent and under-specified items are rejected.

---

## 6. Generation architecture

### 6.1 Separate problem, solution, verifier, and language roles

The same model instance must not generate a row and be its only judge.

Recommended roles:

1. **Problem generator:** creates a structured problem specification.
2. **Deterministic solver:** computes the canonical answer where possible.
3. **Trace teacher:** generates multiple Arabic reasoning traces.
4. **Independent verifier:** checks the problem, answer, and each reasoning step.
5. **Arabic editor/reviewer:** checks native quality without changing mathematical meaning.
6. **Final verifier:** reruns after all edits.

Agreement between two prompts to the same teacher is not independent verification.

### 6.2 Verifier-first problem generation

Preferred sequence:

1. Create a machine-readable latent specification.
2. Solve it programmatically.
3. Prove answer uniqueness when required.
4. Render the prompt in natural MSA.
5. Re-extract quantities/constraints from the rendered Arabic.
6. Verify the rendered prompt still represents the solved specification.
7. Generate reasoning traces with access controlled by task class: solver-backed trace generation may receive the structured derivation; blind-solving traces may receive only the prompt. In either case, provenance records what was visible and verification must catch unsupported copying.

For logic, use Z3 or an equivalent constraint solver.  
For arithmetic/algebra, use exact Python/SymPy execution.  
For competition math, combine symbolic/numeric checks with a separate reasoning verifier.

### 6.3 Multi-trace sampling

- ordinary problem: four separately sampled candidate traces;
- difficult problem: up to eight;
- sixteen only for targeted analysis or hard-item diversity;
- deduplicate traces semantically;
- select a fully correct and sufficient trace within the calibrated length band; use concision only as a tie-breaker;
- optionally retain one genuinely distinct solution method.

Store every raw candidate and its rejection reasons. Do not discard the audit trail.

### 6.4 Step-level verification

Final-answer agreement is necessary but insufficient.

Split each trace into atomic reasoning steps and verify what is executable:

- arithmetic calculation;
- equation transformation;
- use of assumptions;
- constraint satisfaction;
- logical entailment;
- absence of contradiction;
- final answer derivation.

This is a hard gate only for solver-backed arithmetic, algebra, and finite-constraint classes. For unrestricted competition proofs, the verifier may not establish universal correctness; such rows require expert review or quarantine. A generative process verifier can triage hard cases, but it must be calibrated against human labels and cannot override a failed executable check.

### 6.5 Concision pass

After verification:

- remove repeated problem statements;
- remove redundant calculations;
- remove “first/second/finally” filler when it carries no logic;
- remove meta-commentary;
- preserve all necessary derivations;
- rerun step verification after editing.

Never prune by character count alone.

---

## 7. Deduplication, decontamination, and split integrity

### 7.1 Assign split before generating variants

Every problem receives a stable `family_id` before paraphrasing, translation, numeric substitution, or trace generation.

The following belong to one family:

- translations;
- paraphrases;
- numeric substitutions of one template;
- the same equation or proof with renamed variables;
- the same logic graph with renamed entities;
- story-context rewrites;
- difficulty-hiked descendants;
- multiple traces for the same prompt.

All members of a family must stay in one **partition** (`train`, `validation`, or `private_eval`). A train-family may be promoted from RLVR into a later rejection-sampled SFT stage, but it remains train-only, records lineage and promotion reason, is removed from the active RLVR pool for stage-level reporting, and is permanently excluded from all evaluation sets.

### 7.2 Required decontamination stack

1. Unicode and Arabic normalization.
2. Exact normalized hash.
3. Character/word n-gram matching.
4. MinHash/LSH near-duplicate retrieval.
5. Embedding retrieval against all internal splits and external benchmarks.
6. Independent LLM paraphrase adjudication on retrieved candidates.
7. Human adjudication for unresolved high-similarity pairs.

Benchmark matching must include:

- original language;
- known Arabic translations;
- generated paraphrases;
- answer-preserving numeric variants;
- logic graph equivalence where applicable.

### 7.3 Hard split rules

- zero exact prompt overlap across SFT, RLVR, validation, and evaluation;
- zero family overlap between any train partition and validation/private-eval;
- zero SFT/RLVR family overlap in active initial-stage pools; later promoted RLVR→SFT families use the explicit lineage exception above and leave the active RLVR pool;
- zero detected unresolved semantic-overlap candidates against the versioned internal and benchmark registry;
- zero public benchmark test items or translations in training;
- benchmark-derived training data must use train splits only and remain separated from benchmark evaluation;
- evaluation is frozen and registered before full generation.

The current RLVR eval must be retired as a clean benchmark. It may remain as a contaminated diagnostic set with an explicit label. No process can guarantee absence of unknown contamination; the enforceable claim is zero detected unresolved matches against a named registry, model, threshold set, and audit version.

---

## 8. Arabic quality assurance

Arabic quality cannot be reduced to a single purity score.

### 8.1 Automated checks

Measure separately:

- Arabic prose ratio, excluding equations/JSON/variables;
- foreign prose spans;
- dialect markers;
- Unicode/script consistency;
- malformed punctuation and spacing;
- repeated lexical patterns;
- translation-literalness indicators;
- terminology consistency;
- prompt readability;
- answer/trace grammatical agreement where applicable.

Before applying numeric thresholds, the Data and Arabic QA leads must freeze:

- the tokenizer and Unicode normalizer;
- which spans are excluded as math/JSON/code;
- dialect lexicon/classifier version;
- literal-translation classifier or rubric;
- calibration set with native-review labels;
- threshold sensitivity report.

Acceptance targets:

- adjusted Arabic purity per row ≥0.85;
- batch median ≥0.95;
- batch p5 ≥0.90;
- zero unexplained English sentences;
- zero unlabeled dialectal traces.

These thresholds are initial policy choices and must be recalibrated against the pilot's native-review results; they are not universal Arabic-quality constants.

### 8.2 Native review

Use native Arabic reviewers with STEM competence or paired Arabic/STEM reviewers.

Review:

- natural MSA;
- clarity;
- mathematical fidelity;
- age/grade appropriateness;
- ambiguity;
- unnatural avoidance of the result;
- template repetition;
- translated idioms or English syntax.

Any linguistic edit triggers mathematical re-verification.

---

## 9. Human QA protocol

### 9.1 Rubric

Each reviewed row receives explicit labels for:

- problem validity;
- determinacy/uniqueness;
- canonical answer correctness;
- each reasoning step;
- completeness;
- concision;
- Arabic naturalness;
- format compliance;
- contamination suspicion;
- severity: clean / minor / major / critical.

### 9.2 Sampling

Pilot:

- review at least 300 **accepted** rows for accepted-data error estimation;
- review a separate stratified sample of at least 100 rejected rows to audit false rejections and rejection reasons;
- stratify by domain, difficulty, source, teacher, verifier, and length;

Production:

- review at least 300 accepted rows or 2% of every batch, whichever is larger, when the critical-error upper-bound gate is applied;
- oversample hard items, verifier disagreements, low Arabic scores, and length outliers;
- double-annotate at least 100 accepted rows per release (or 30% when the release has fewer than 334 rows);
- adjudicate all critical disagreements.

### 9.3 Statistical reporting

- report raw counts and one-sided 95% Wilson confidence intervals using the accepted-row sample;
- report inter-annotator agreement;
- target Cohen's κ ≥0.80 for binary correctness and weighted κ ≥0.80 for ordinal severity;
- target κ ≥0.70 for language naturalness;
- do not report only an average “quality score”;
- do not populate every row with `quality.score = 1.0`.

With zero critical failures in 300 independently reviewed accepted samples, the approximate one-sided 95% upper bound is around 1%. If any critical failure occurs, compute the interval and expand the sample; do not claim the <1% bound automatically. Twenty unscored examples cannot support a production claim.

---

## 10. Required row schema and provenance

The release contains four distinct versioned record types with separate JSON Schemas:

1. **Problem record:** prompt, family, answer specification, provenance, and verifier.
2. **SFT trace record:** problem reference plus a verified `<think>/<answer>` response.
3. **RLVR prompt record:** problem reference and executable verifier; no teacher response.
4. **Evaluation record:** evaluation-owned problem/verifier with restricted labels and access.

Response-format, reasoning-step, and response-length gates apply only to SFT traces and sampled model rollouts—not to response-empty RLVR prompt rows.

Minimum problem-record metadata:

```json
{
  "problem_id": "stable-id",
  "family_id": "stable-family-id",
  "parent_id": null,
  "domain": "word_arithmetic|school_math|competition_math|logic_constraints",
  "subdomain": "fractions",
  "skills": ["ratio", "division"],
  "prompt": "Arabic prompt",
  "answer_spec": {
    "type": "integer|rational|decimal_exact|decimal_approx|symbolic|logic_json",
    "canonical": "canonical answer string",
    "tolerance": null,
    "unit": null,
    "normalizer_version": "answer-v1",
    "verifier_artifact": "verifiers/id.json"
  },
  "ground_truth_structured": null,
  "verifier_type": "python|sympy|z3|hybrid",
  "verifier_version": "git-sha",
  "verifier_result": true,
  "difficulty_teacher": 4,
  "difficulty_student_pass_at_n": 0.375,
  "source_type": "native_ar|translated|synthetic_from_spec|derived",
  "source_license": "identifier",
  "partition": "train|validation|private_eval",
  "stage_eligibility": ["sft", "rlvr"],
  "generator_model": "model-id",
  "generator_prompt_version": "hash",
  "generation_seed": 123,
  "generation_temperature": 0.7,
  "pipeline_version": "git-sha"
}
```

Minimum per-trace metadata:

```json
{
  "trace_id": "stable-trace-id",
  "problem_id": "stable-id",
  "response": "<think>\n...\n</think>\n<answer>...</answer>",
  "token_count": 240,
  "arabic_metrics": {},
  "step_verification": [],
  "selected": true,
  "rejection_reasons": [],
  "human_labels": []
}
```

The data team must deliver:

- immutable raw generation archive;
- all candidate traces;
- accepted SFT dataset;
- RLVR prompt/verifier pool;
- frozen private evaluation manifest;
- quarantine and rejection dataset;
- contamination registry;
- human-QA report;
- data card and license report;
- checksums and reproducible generation manifests;
- exact generator/verifier prompts and code versions.

Raw candidates are retained only under the source licenses and organizational privacy policy. The release manifest must define retention duration, access roles, deletion procedure, and whether prompts may contain personal or proprietary material. Public releases include only artifacts whose licenses permit redistribution.

---

## 11. Acceptance gates

### 11.1 Problem-record hard gates

Every accepted problem must pass:

- exact schema parse;
- canonical answer parse;
- executable answer verifier;
- unique answer where required;
- prompt Arabic gate;
- no detected unresolved contamination match against the frozen registry;
- family and partition assignment;
- complete provenance and licensing fields.

### 11.2 SFT-trace hard gates

Every accepted SFT trace must pass:

- exact trace schema and tag grammar;
- non-empty substantive think under the frozen rubric;
- answer matches its problem's `answer_spec`;
- all executable critical steps pass;
- unsupported steps in non-executable classes receive expert approval;
- no incomplete/truncated sentence under automated detection and reviewer rubric;
- no unsupported answer copying;
- adjusted Arabic purity gate;
- response length policy;
- no truncation.

### 11.3 RLVR-prompt hard gates

Every accepted RLVR row must pass:

- exact RLVR schema;
- no stored training response;
- executable answer verifier and canonicalizer;
- student `pass@8` calibration or explicit `deferred/mastered` status;
- zero family overlap with validation/private-eval;
- provenance, licensing, and contamination gates.

### 11.4 Evaluation-item hard gates

Every evaluation item must pass the problem gates plus independent Evaluation-team approval. Labels and verifier outputs remain access-controlled and are never sent to generation models.

One failed hard gate rejects the row.

### 11.5 Per-release hard gates

- 100% accepted-row answer verification;
- 100% format validity for SFT trace records and named sampled-model rollout evaluations, not response-empty Problem/RLVR records;
- zero exact and family split overlap;
- zero detected unresolved contamination candidates against the frozen registry;
- latent duplicate-family rate below 1% (different `family_id` values adjudicated as the same family);
- clipped rate below 1% in the named student rollout evaluation (not as a static-data property);
- Arabic median ≥0.95 and p5 ≥0.90;
- 100% domain, skill, and empirical-difficulty metadata coverage;
- release, QA report, canonicalizer, verifier, and corpus version hashes agree;
- one-sided 95% upper confidence bound for human major+critical error below 2%;
- human critical-error upper bound below 1%;
- reviewer agreement thresholds met;
- no single uncontrolled template family dominates the batch.

Do not average correctness, format, Arabic, and contamination into one pass score. Critical defects cannot be canceled by high scores elsewhere.

---

## 12. Pilot-to-production rollout

### Ownership and sign-off

- **Data lead:** schemas, generation, provenance, licensing, deterministic verifier packaging, family IDs, deduplication, Arabic automated QA, release manifests.
- **Arabic/STEM QA lead:** review rubric, reviewer calibration, adjudication, human-QA statistics.
- **Evaluation lead:** private-eval creation, benchmark registry, contamination sign-off, label access, evaluation protocol.
- **Training lead:** chat template, adapter loading, reward semantics, rollout settings, clipping measurement, SFT/RL metrics.
- **Infrastructure lead:** CUDA stress test, memory profile, checkpoint/resume, reproducible environment.

Phase 0 requires a signed manifest from all five owners. Data-team delivery does not certify training stability, and the Training team does not control private evaluation labels.

### Phase 0 — infrastructure

Complete the frozen contracts, runtime invariants, clean evaluation, verifier code, and GPU stress test.

**Stop condition:** any mismatch between offline and training behavior.

### Phase 1 — 500-family pilot

Generate 500 families × four traces. Run the complete verification, deduplication, Arabic QA, and human-review pipeline.

Train:

- short SFT;
- greedy format/correctness probe;
- clean held-out evaluation.

**Go condition:** all applicable Problem/SFT and release gates in Section 11 pass; RLVR/evaluation gates apply only when those record types are included.

### Phase 2 — 2,000-family cold start

Produce 3,000–4,000 accepted traces. Train from the base model, prove adapter loading, and evaluate on the frozen clean suite.

Required before RL:

- format ≥99% under greedy/low-temperature decoding;
- Arabic compliance ≥95%;
- clipped rate <1%;
- empty-think rate 0%;
- easy-task greedy correctness ≥60%;
- medium-task pass@8 ≥30%;
- multi-hour training stress test passes.

### Phase 3 — limited RLVR

Use 1,000–3,000 frontier-calibrated prompts with four rollouts initially. Increase to eight only when memory and throughput are stable.

Track:

- reward variance per prompt;
- zero-variance group rate;
- correctness by domain/difficulty;
- format, Arabic, length, entropy, and KL;
- clipping;
- verifier loopholes.

### Phase 4 — rejection-sampled second SFT

Curate successful trajectories and corrected informative failures. Retrain SFT and evaluate before a second RL stage.

### Phase 5 — controlled full expansion

Expand toward:

- 7,000–10,000 accepted SFT traces;
- 6,000–10,000 unique RLVR prompts;
- 1,000–2,000 private eval prompts.

Scale in tranches. Every tranche must produce a statistically credible gain on clean evaluation.

Default tranche:

- +1,000 unique train families;
- fixed training-token budget comparison against the previous tranche;
- at least two training seeds for tranche decisions and three for the final release;
- primary metric: macro-average exact/executable pass@1 correctness across the four domains;
- secondary metrics: per-domain pass@1, pass@8, format, Arabic compliance, clipping, and calibration;
- report paired item-bootstrap 95% confidence intervals and seed dispersion.

“Credible gain” means the primary metric improves and its paired bootstrap interval excludes zero, with no material regression (>2 absolute points) in any core domain. If compute cannot support this protocol, label the tranche exploratory rather than production-approved.

Stop scaling when:

- two consecutive data increments produce <1 percentage point gain;
- correctness regresses;
- hard data harms medium-tier performance;
- reward hacking increases;
- contamination or human error gates fail.

---

## 13. Changes required outside the data team

The data team should not be held responsible for these engineering tasks, but generation should not proceed without them:

1. Keep the chat-template assertion that generation begins before `<think>`.
2. Keep strict adapter loading and tensor/hash verification.
3. Replace exact-string-only symbolic scoring with executable equivalence or restrict answer types.
4. Align length reward with accepted trace distributions and difficulty.
5. Remove penalties against naturally derived final values in `<think>`.
6. Raise completion capacity while reducing rollout count to fit memory.
7. Start RL with four rollouts, not sixteen, on the 8 GB GPU.
8. Add checkpoint/resume and CUDA stress tests.
9. Retire contaminated eval and report only private family-disjoint results.
10. Add a true SFT evaluation before GRPO.
11. Log per-domain/per-difficulty correctness, not only aggregate reward.
12. Fail the run if prompt rendering, adapter loading, or verifier loading differs from the audited configuration.

---

## 14. Definition of success

Success is not “the generator produced 50,000 rows.”

Success means:

- the student learned format without reward shaping tricks;
- correctness improves on a private family-disjoint Arabic evaluation;
- RL groups contain usable reward variance;
- reasoning is concise, natural, and executable;
- no benchmark leakage inflates results;
- every accepted answer has a reproducible verifier;
- human reviewers confirm low critical-error rates;
- gains survive retraining with a different seed;
- the training system runs long enough to measure convergence.

The immediate recommendation is:

> Build and validate a 500-family pilot, then a 2,000-family cold-start corpus. Do not authorize full-scale expansion until the student passes a clean SFT gate and a stable limited-RLVR run. Quality-controlled tranches are more valuable than a large irreversible generation dump.

---

## Research basis

Internal evidence used in this proposal:

- `runs/RESEARCH_DIAGNOSIS.md`
- `runs/sft_v4/format_probe_results.json`
- `runs/grpo_v11_smoke/run.out.log` and `run.err.log`
- `docs/data_v2_dspy_gepa/qa_report_ship_gate.json`
- `docs/data_v2_dspy_gepa/DATA_QUALITY_REPORT.md`
- `data/human_review_sample_20.jsonl`
- the four active `data/arabic_reasoning_{coldstart,rlvr}_{train,eval}.jsonl` files

Internal counts are configuration-specific observations from the 2026-07-18 audit, not universal conclusions about GRPO or Arabic reasoning.

- DeepSeek-R1: cold-start readable reasoning, RL, rejection sampling, second SFT, and explicit warning about empty-think bypass  
  https://arxiv.org/html/2501.12948

- OpenMathInstruct-2: strong-teacher synthesis, question diversity, concise solution formats, and embedding + LLM decontamination  
  https://arxiv.org/abs/2410.01560

- Data Recipes for Reasoning Models: exact deduplication and multi-trace sampling; quality/diversity ablations  
  https://arxiv.org/abs/2506.04178

- Think, Prune, Train: correctly pruned synthetic traces improve a 2B model; scaling may plateau  
  https://arxiv.org/abs/2504.18116

- SAND-Math: semantic deduplication, benchmark decontamination, solver-based difficulty, and difficulty hiking  
  https://arxiv.org/abs/2507.20527

- ThinkPRM: filtered synthetic step-verification data and generative process verification  
  https://arxiv.org/abs/2504.16828

- 3LM: native Arabic STEM sources, synthetic Arabic STEM, and human-in-the-loop translation validation  
  https://aclanthology.org/2025.arabicnlp-main.4/

- ACL 2024 contamination survey: detection, impact, and mitigation of benchmark contamination  
  https://aclanthology.org/2024.findings-acl.951/

