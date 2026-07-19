# RLVR Codebase Bug Report

**Audit session:** `d187aa`  
**Date:** 2026-07-19  
**Scope:** `src/rlvr/`, `src/rlvr_pipeline/`, `src/rlvr_sota/`, `src/rlvr_contracts/`, `scripts/`  
**Method:** AST syntax sweep (49 files) + runtime hypothesis checks (no GPU/TRL)

Use the **Status** column to track fixes. Values: `open` | `fixed` | `wontfix` | `needs-check`.

---

## Quick comparison checklist

| ID | Severity | Short title | Status | Primary location |
|----|----------|-------------|--------|------------------|
| A | critical | Logic `True == 1` / `False == 0` | open | `src/rlvr_contracts/verifiers.py` `_flat_logic_equal` |
| B | high | Legacy math `float(True)` accepts `"1"` | open | `src/rlvr_contracts/verifiers.py` `verify_legacy` |
| C | high | `replay_buffer` never replaces groups | open | `src/rlvr_pipeline/failure_mining_trainer.py` |
| D | high | Curriculum ignores `max_steps` | open | `failure_mining_trainer.attach_curriculum_sampler` |
| E | high | Language weight 0.2 vs 0.05 | open | `reward_composer.py` vs `rewards.py` / YAML |
| F | high | Empty `sample_id` CRPS key collision | open | `_crps_failure_key` |
| G | medium | RL-ZVP skips all-correct flat groups | open | `_process_failure_mining` direct_scoring |
| H | medium | Failure bank retrieved but unused | open | `src/rlvr/pipeline.py` |
| SYN | medium | UTF-8 BOM breaks `ast.parse` | open | `data.py` / `rewards.py` (pipeline + sota) |
| I | low | Dataset silent drops (data file absent at audit) | needs-check | `src/rlvr_pipeline/data.py` |

---

## Environment notes (not code bugs)

At audit time this Python env lacked:

- `pytest`
- `PyYAML` (`yaml`)
- `numpy`

So the full unit suite could not run. Re-verify after installing project deps with:

```bash
set PYTHONPATH=src
pytest tests/ -q
```

---

## BUG-A — Logic verifier treats bool as int

**Severity:** critical  
**Status:** open  
**Where:** `src/rlvr_contracts/verifiers.py` (~33–49)

**Why:** `bool` is a subclass of `int` in Python.  
`isinstance(True, (int, float))` is true, so the numeric branch runs and `True`/`1` compare equal. The early `type(a) is not type(b)` branch is a no-op (`pass`).

**Runtime evidence (audit):**

| Check | Result |
|-------|--------|
| `_flat_logic_equal(True, 1)` | `True` |
| `_flat_logic_equal(False, 0)` | `True` |
| `_flat_logic_equal({"ok": True}, {"ok": 1})` | `True` |
| `verify_answer('{"ok": 1}', logic_json GT ok=True)` | score `1.0` |

**Fixed when:**

```text
_flat_logic_equal(True, 1) → False
_flat_logic_equal({"ok": True}, {"ok": 1}) → False
_flat_logic_equal(True, True) → True
_flat_logic_equal(1, 1.0) → True   # int/float still OK
```

**Suggested fix:** Treat `bool` separately (require same type), or exclude bool from the numeric `isinstance` branch.

---

## BUG-B — Legacy math coerces bool GT via `float()`

**Severity:** high  
**Status:** open  
**Where:** `src/rlvr_contracts/verifiers.py` `verify_legacy` (~244–251)

**Why:** `float(True) == 1.0`, so answer `<answer>1</answer>` scores full credit against ground truth `True`. Serialized string `"true"` does **not** take this path and correctly fails numeric parse.

**Runtime evidence:**

| GT | Completion answer | Score |
|----|-------------------|-------|
| `True` (bool) | `1` | `1.0` (bug) |
| `"true"` (str) | `1` | `0.0` |
| `1` (int) | `1` | `1.0` (intended) |

**Fixed when:** bool GT does not score via numeric float path (reject, or stringify like `"true"`/`"false"` first).

**Also affects callers that pass raw bools:** research `pipeline.py`, `difficulty_labeler.py` (when not serialized).

---

## BUG-C — `replay_buffer` strategy is a dead path

**Severity:** high  
**Status:** open  
**Where:**

- Implemented: `src/rlvr/failure_mining.py` `replace_ineffective_group`
- Trainer: `src/rlvr_pipeline/failure_mining_trainer.py` (~235–252)  
  (identical copy in `src/rlvr_sota/failure_mining_trainer.py`)

**Why:** Config allows `zero_variance_strategy: "replay_buffer"`. Trainer only **pushes** effective groups into the buffer. It never calls `replace_ineffective_group`, so ineffective (zero-variance) groups still train with zero advantages while the buffer grows unused for replacement.

**Runtime / static evidence:**

| Check | Result |
|-------|--------|
| `def replace_ineffective_group` exists | yes |
| Trainer contains `"replay_buffer"` branch | yes |
| Trainer calls `replace_ineffective_group` | **no** |

**Fixed when:** ineffective groups are swapped from the buffer (or the config option is removed / documented as bank-only).

---

## BUG-D — Curriculum `total_steps` ignores `max_steps`

**Severity:** high  
**Status:** open  
**Where:** `attach_curriculum_sampler` in  
`src/rlvr_pipeline/failure_mining_trainer.py` (~65–68)

```python
total_steps = steps_per_epoch * config.num_train_epochs
```

**Why:** Smoke / CLI runs often set `--max-steps 30`. Curriculum still schedules as if training for full epoch×dataset length (often ≫30). Hard bucket weight near step 29 stays ~0, so CRPS (hard-only) never activates.

**Runtime evidence:**

| Call | `hard` weight |
|------|---------------|
| `compute_stage_sampling_weights(29, 1000)` | ~`6.2e-05` |
| `compute_stage_sampling_weights(29, 30)` | ~`0.76` |
| Trainer reads `config.max_steps` for curriculum | **no** |

**Fixed when:** `total_steps = config.max_steps` (when set), else epoch formula; at smoke step 29, hard weight is meaningful.

---

## BUG-E — Language reward weight mismatch

**Severity:** high  
**Status:** open  
**Where:**

| Source | Language weight |
|--------|-----------------|
| `src/rlvr/reward_composer.py` `W_LANGUAGE` / `compose_reward` | **0.2** |
| `src/rlvr_pipeline/rewards.py` `DEFAULT_REWARD_WEIGHTS[2]` | **0.05** |
| `configs/qwen_4b_qlora.yaml` `reward_weights[2]` | **0.05** |
| `configs/qwen_4b_smoke_v11.yaml` | **0.10** (override; intentional) |

**Why:** Research/composite path and TRL multi-reward path disagree. Docstring of `reward_composer` still claims `0.2 * language`.

**Runtime evidence:**

```text
compose weights:  [0.6, 0.2, 0.2, 0.5, 0.3, 0.15]
DEFAULT / YAML:   [0.6, 0.2, 0.05, 0.5, 0.3, 0.15]
bug_confirmed: True
```

**Fixed when:** `W_LANGUAGE == DEFAULT_REWARD_WEIGHTS[2]` (pick one canonical value and update docstring/YAML together).

---

## BUG-F — CRPS failure keys collide when `sample_id` empty

**Severity:** high  
**Status:** open  
**Where:** `_crps_failure_key` in `failure_mining_trainer.py` (~135–140)

```python
return f"{sample_id}:{puzzle_type}:{difficulty}"
```

**Why:** Loader can leave `sample_id=""` when both `problem_id` and `id` are missing. All hard math samples share `":math:hard"` → shared failure counters, wrong hint escalation, premature resets.

**Runtime evidence:**

| Inputs | Key |
|--------|-----|
| empty id, math, hard | `:math:hard` |
| another empty id, math, hard | `:math:hard` (collision) |
| `sample_id=q1`, math, hard | `q1:math:hard` |

**Fixed when:** empty `sample_id` fails closed, or key includes prompt hash / row index.

---

## BUG-G — RL-ZVP skips all-correct zero-variance groups

**Severity:** medium  
**Status:** open  
**Where:** `_process_failure_mining` `direct_scoring` branch (~204–227)

**Why:** Gate requires correctness ≈ 0:

```python
is_zero_variance_group(group_correctness) and abs(group_correctness[0]) < 1e-6
```

Library / design also covers all-correct → `(1 - confidence)`, but trainer never enters that path. All-correct flat groups keep zero GRPO advantages.

**Fixed when:** all-correct zero-variance groups get `(1 - conf)` (or equivalent) advantages.

---

## BUG-H — Failure bank retrieved but unused

**Severity:** medium  
**Status:** open  
**Where:** `src/rlvr/pipeline.py` (~52–55)

```python
if config.curriculum_stage == "hard":
    banked = retrieve_banked_failures("hard")
else:
    banked = []
```

**Why:** `banked` is never mixed into rollouts. Banking still increments counters; hard-stage replay is a no-op.

**Fixed when:** banked failures are interleaved into hard-stage prompts, or the retrieve call is removed.

---

## BUG-SYN — UTF-8 BOM in four source files

**Severity:** medium (tooling); Python import usually OK  
**Status:** open  
**Where (first bytes `EF BB BF`):**

- `src/rlvr_pipeline/data.py`
- `src/rlvr_pipeline/rewards.py`
- `src/rlvr_sota/data.py`
- `src/rlvr_sota/rewards.py`

**Why:** `ast.parse` on raw text fails with `invalid non-printable character U+FEFF`. Runtime `import` often still works (tokenizer strips BOM).

**Fixed when:** files saved as UTF-8 without BOM; `ast.parse` succeeds on all `src/**/*.py`.

---

## BUG-I — Silent sample drops in loader

**Severity:** medium / low  
**Status:** needs-check (dataset file missing at audit)  
**Where:** `src/rlvr_pipeline/data.py` (~131–161)

**Why:** `continue` on mmlu / missing GT / failed legacy infer / missing `answer_spec` with no count or warning. Corpora can shrink quietly.

**Audit note:** `data/arabic_reasoning_rlvr.jsonl` was not present, so drop rates were not measured.

**Fixed when:** loader logs drop counts (or asserts min keep rate); compare `len(jsonl)` vs `len(dataset)`.

---

## Related / lower priority (not fully runtime-proven this session)

| Item | Notes |
|------|--------|
| Dual `rlvr_pipeline` / `rlvr_sota` | Shared modules identical; drift risk if only one is edited |
| Format scoring strict vs partial | `compose_reward` strict; training `format_reward_func` allows partial ≤0.30 |
| Logic `puzzle_type` collapsed to `"logic"` | CRPS related-trace matching may cross puzzle subtypes |
| Zero-variance eps inconsistency | Exact `max==min` vs `(max-min)<1e-6` across modules |

---

## How to re-check after fixes

From repo root (`PYTHONPATH=src`):

```python
from rlvr_contracts.verifiers import _flat_logic_equal, verify_legacy
from rlvr.reward_composer import W_LANGUAGE
from rlvr.curriculum import compute_stage_sampling_weights as w

# A
assert _flat_logic_equal(True, 1) is False
assert _flat_logic_equal({"ok": True}, {"ok": 1}) is False

# B
assert verify_legacy("<think>x</think>\n<answer>1</answer>", True, "math") < 0.5

# D — smoke-scale schedule should reach hard by step 29
assert w(29, 30)["hard"] > 0.5

# E — align compose with pipeline default (canonical target: 0.05)
assert abs(W_LANGUAGE - 0.05) < 1e-9
```

Also:

```bash
rg "replace_ineffective_group" src/rlvr_pipeline/failure_mining_trainer.py
# expect a call site after fix (C)

rg "\bbanked\b" src/rlvr/pipeline.py
# expect use beyond assign, or retrieve removed (H)
```

---

## Priority fix order

1. **A** — bool/int logic equality  
2. **B** — legacy bool math coercion  
3. **C** — wire or remove `replay_buffer` replacement  
4. **D** — curriculum `total_steps` from `max_steps`  
5. **E** — align `W_LANGUAGE` with TRL/YAML  
6. **F** — CRPS key uniqueness  
7. **G / H / SYN** — as follow-ups  

---

## Change log

| Date | Note |
|------|------|
| 2026-07-19 | Initial report from session `d187aa` runtime audit |
