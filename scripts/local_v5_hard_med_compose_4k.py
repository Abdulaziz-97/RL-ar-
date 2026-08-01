#!/usr/bin/env python3
"""Generate hard/medium RLVR top-up and compose a 2000-row v5 set.

Flow:
  1) Pull pass@8-stamped shards from Vast (band-aware keep).
  2) Programmatic hard/medium templates (no teacher).
  3) Compose 2000: pass@8 non-mastered (easy/deferred/medium/hard) + new hard/med.
  4) Optional Flash RQA if DEEPSEEK_API_KEY available.
  5) Dedup vs coldstart; write candidates_v5 + rlvr_v5 locally.

Non-mastered pass@8 bands (~401) plus generated fill; final set is **500 each**
of easy / deferred / medium / hard (2000 total). No mastered rows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(r"C:\Users\Azooo\arabic-reasoning-rlvr-sota\RL-ar-")
DATA1 = REPO / "data_1"
WORK = DATA1 / "outputs" / f"rlvr_hard_med_topup_{time.strftime('%Y%m%d_%H%M%S')}"
LOG = REPO / "outputs" / "local_v5_hard_med_4k.log"
COLD = REPO / "data" / "arabic_reasoning_coldstart_v5.jsonl"
CAND = REPO / "data" / "arabic_reasoning_rlvr_candidates_v5.jsonl"
V5 = REPO / "data" / "arabic_reasoning_rlvr_v5.jsonl"
PASS8_LOCAL = WORK / "pass8_stamped.jsonl"
STATS = REPO / "data" / "arabic_reasoning_rlvr_v5_compose_STATS.json"

TARGET_TOTAL = 2000
BAND_QUOTAS = {"easy": 500, "deferred": 500, "medium": 500, "hard": 500}
SKIP_RQA = os.environ.get("SKIP_RQA", "1") == "1"
REUSE_WORK = os.environ.get("REUSE_WORK", "")
BALANCED_ONLY = os.environ.get("BALANCED_ONLY", "0") == "1"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    wakeb = Path(r"C:\Users\Azooo\Project_3_wakeb\.env")
    if wakeb.is_file():
        for line in wakeb.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("Deep_seek_api_key"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["DEEPSEEK_API_KEY"] = key
                return key
    return None


def run_py(args: list[str], env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(DATA1), str(DATA1 / "src"), str(DATA1 / "vendor")]
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    log("RUN " + " ".join(args))
    p = subprocess.run([sys.executable, *args], env=env, cwd=str(REPO))
    if p.returncode != 0:
        raise RuntimeError(f"exit={p.returncode}: {' '.join(args)}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prompt_key(row: dict) -> str:
    return (row.get("prompt") or "").strip()


def num_steps(row: dict) -> int:
    try:
        return int((row.get("metadata") or {}).get("num_steps") or 0)
    except (TypeError, ValueError):
        return 0


def band_of(row: dict) -> str:
    b = str((row.get("empirical_difficulty") or {}).get("band") or "")
    if b == "hard_diagnostic":
        return "hard"
    return b


def is_deferred_heur(row: dict) -> bool:
    """Hardest bucket — maps to pass@8 deferred (≈0–1/8 passes)."""
    ns = num_steps(row)
    dom = str(row.get("domain") or "")
    fam = str((row.get("metadata") or {}).get("template_family") or "")
    if ns >= 7:
        return True
    if fam.startswith("_hard_") or fam.startswith("gsm_stacked") or "hard_" in fam:
        return True
    if dom == "logic" and ns >= 4:
        tag = (row.get("metadata") or {}).get("difficulty_tag")
        return tag == "hard"
    return False


def is_easy_heur(row: dict) -> bool:
    if is_deferred_heur(row) or is_hard(row) or is_medium(row):
        return False
    return num_steps(row) <= 2


def compose_band(row: dict) -> str:
    """Empirical pass@8 band when present, else heuristic curriculum band."""
    b = band_of(row)
    if b in BAND_QUOTAS:
        return b
    if is_deferred_heur(row):
        return "deferred"
    if is_hard(row):
        return "hard"
    if is_medium(row):
        return "medium"
    return "easy"


def stamp_compose_band(row: dict, band: str) -> dict:
    """Record intended training band for rows without pass@8 calibration."""
    out = dict(row)
    if not band_of(out):
        md = dict(out.get("metadata") or {})
        md["compose_band"] = band
        md.setdefault("difficulty_tag", band)
        out["metadata"] = md
        emp = dict(out.get("empirical_difficulty") or {})
        emp.setdefault("band", band)
        emp["release_band"] = band
        out["empirical_difficulty"] = emp
    return out


def is_hard(row: dict) -> bool:
    ns = num_steps(row)
    dom = str(row.get("domain") or "")
    tag = (row.get("metadata") or {}).get("difficulty_tag") or row.get("difficulty_tag")
    if ns >= 5:
        return True
    if dom == "logic" and ns >= 3:
        return True
    if tag == "hard" and ns >= 3:
        return True
    return False


def is_medium(row: dict) -> bool:
    if is_hard(row):
        return False
    ns = num_steps(row)
    tag = (row.get("metadata") or {}).get("difficulty_tag") or row.get("difficulty_tag")
    if tag == "medium" and ns >= 2:
        return True
    return 3 <= ns <= 4


def pull_pass8_stamped() -> list[dict]:
    """Fetch calibrated shards from Vast into one local jsonl."""
    import paramiko

    # Reuse a prior SCP if present (avoids re-download every rerun).
    prior = sorted(
        (DATA1 / "outputs").glob("rlvr_hard_med_topup_*/cand_shard0.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    local0 = WORK / "cand_shard0.jsonl"
    local1 = WORK / "cand_shard1.jsonl"
    WORK.mkdir(parents=True, exist_ok=True)
    if prior:
        p0 = prior[0]
        p1 = p0.with_name("cand_shard1.jsonl")
        if p0.is_file() and p1.is_file():
            import shutil

            log(f"reuse pass8 shards from {p0.parent.name}")
            shutil.copy2(p0, local0)
            shutil.copy2(p1, local1)
        else:
            prior = []
    if not prior:
        remote0 = "/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8/cand_shard0.jsonl"
        remote1 = "/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8/cand_shard1.jsonl"
        key = paramiko.Ed25519Key.from_private_key_file(
            str(Path.home() / ".ssh" / "id_ed25519"), password="aziz"
        )
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(
            "8.243.214.78",
            port=49198,
            username="root",
            pkey=key,
            timeout=60,
            allow_agent=False,
            look_for_keys=False,
        )
        sftp = c.open_sftp()
        for rem, loc in ((remote0, local0), (remote1, local1)):
            log(f"SCP {rem} -> {loc}")
            sftp.get(rem, str(loc))
        sftp.close()
        c.close()
    rows = read_jsonl(local0) + read_jsonl(local1)
    # strip top-level difficulty_tag for schema safety
    cleaned = []
    for r in rows:
        r = dict(r)
        tag = r.pop("difficulty_tag", None)
        if tag is not None:
            md = dict(r.get("metadata") or {})
            md.setdefault("difficulty_tag", tag)
            r["metadata"] = md
        cleaned.append(r)
    write_jsonl(PASS8_LOCAL, cleaned)
    log(f"pass8_stamped={len(cleaned)} bands={Counter(band_of(r) for r in cleaned)}")
    return cleaned


def _parse_int_gt(gt) -> int | None:
    """Accept int, digit string, or 'x = N' math_comp forms."""
    import re

    if isinstance(gt, bool):
        return None
    if isinstance(gt, int):
        return gt
    if isinstance(gt, float) and gt == int(gt):
        return int(gt)
    if isinstance(gt, str):
        s = gt.strip()
        m = re.match(r"(?i)^x\s*=\s*(-?\d+)\s*$", s)
        if m:
            return int(m.group(1))
        try:
            return int(s)
        except ValueError:
            return None
    return None


def sample_to_row(sample, *, seed: int) -> dict | None:
    """Convert programmatic Sample → RLVR candidate row (no teacher / no API)."""
    sys.path[:0] = [
        str(DATA1),
        str(DATA1 / "src"),
        str(DATA1 / "vendor"),
        str(REPO / "src"),
    ]
    from rlvr_synth.quality.ids import content_family_id, content_problem_id

    gt = sample.ground_truth
    if isinstance(gt, dict):
        answer_spec = {
            "type": "logic_json",
            "canonical": json.dumps(gt, ensure_ascii=False, sort_keys=True),
            "ground_truth_structured": gt,
        }
        canon = answer_spec["canonical"]
        store_gt = gt
    else:
        ival = _parse_int_gt(gt)
        if ival is None:
            return None
        answer_spec = {
            "type": "integer",
            "canonical": str(ival),
            "ground_truth_structured": ival,
        }
        canon = answer_spec["canonical"]
        store_gt = ival

    pid = content_problem_id(
        domain=sample.domain,
        prompt=sample.prompt,
        answer_canonical=canon,
        partition="rlvr_train",
        seed=seed,
    )
    fid = content_family_id(
        domain=sample.domain,
        prompt=sample.prompt,
        template_family=str((sample.meta_extra or {}).get("template_family") or sample.domain),
        seed=seed,
    )
    ns = int(sample.num_steps)
    meta = {
        "ground_truth": store_gt,
        "num_steps": ns,
        "template_family": (sample.meta_extra or {}).get("template_family"),
        "oracle_ground_truth": canon,
        "solution_steps": list(sample.solution_steps or []),
        "seed": seed,
        **{k: v for k, v in (sample.meta_extra or {}).items() if k != "template_family"},
    }
    # ensure curriculum tag when missing
    if not meta.get("difficulty_tag"):
        if ns >= 5 or (sample.domain == "logic" and ns >= 3):
            meta["difficulty_tag"] = "hard"
        elif ns >= 3:
            meta["difficulty_tag"] = "medium"
        else:
            meta["difficulty_tag"] = "easy"
    return {
        "problem_id": pid,
        "family_id": fid,
        "partition": "rlvr_train",
        "domain": sample.domain,
        "prompt": sample.prompt,
        "answer_spec": answer_spec,
        "verifier_type": "python",
        "verifier_version": "rlvr-contracts-verifiers-v1",
        "provenance": {
            "source": "programmatic_hard_med_topup",
            "seed": seed,
            "reverse_qa": False,
        },
        "licensing": {"license": "internal"},
        "lineage": {
            "generator_version": "rlvr_synth_hard_med_v5",
            "seed": seed,
            "template_family": meta.get("template_family"),
        },
        "metadata": meta,
        "response": "",
    }


def generate_pool() -> Path:
    """Generate samples to fill per-band deficits vs pass@8 non-mastered."""
    sys.path[:0] = [
        str(DATA1),
        str(DATA1 / "src"),
        str(DATA1 / "vendor"),
        str(DATA1 / "vendor" / "synth"),
        str(REPO / "src"),
    ]
    import re
    import random
    import programmatic as prog

    out = WORK / "gen" / "release_corpora" / "rlvr_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    cold_prompts = [prompt_key(r) for r in read_jsonl(COLD) if prompt_key(r)]
    stamped = read_jsonl(PASS8_LOCAL)
    stamped_prompts = [prompt_key(r) for r in stamped if prompt_key(r)]
    seen: set[str] = set(cold_prompts + stamped_prompts)

    pass8_by_band: Counter[str] = Counter()
    for r in stamped:
        b = band_of(r)
        if b in BAND_QUOTAS:
            pass8_by_band[b] += 1

    band_fns = {
        "easy": [
            prog._gsm_shop_change,
            prog._gsm_fraction_of,
            prog._gsm_remaining_after,
            prog._gsm_bus_split,
            prog._gsm_salary_save,
            prog._gsm_garden_rows,
            prog._math_percent,
            prog._math_speed,
            prog._math_area_rect,
            prog._math_unit_price,
            prog._logic_color_objects,
        ],
        "medium": [
            prog._gsm_multi_buy,
            prog._gsm_trip_days,
            prog._gsm_ages,
            prog._gsm_ratio_share,
            prog._gsm_work_rate,
            prog._logic_who_has,
            prog._logic_seating_left_right,
            prog._logic_schedule_slots,
            prog._mc_quadratic_root_sum,
            prog._mc_arithm_seq,
            prog._mc_lcm_product,
        ],
        "hard": [
            prog._gsm_stacked_ops,
            prog._logic_ordering,
            prog._logic_truth,
            prog._mc_gcd_style,
            prog._mc_power_diff,
            prog._mc_digit_sum_constraint,
        ],
        "deferred": [
            prog._hard_books_library,
            prog._hard_factory_shipment,
            prog._hard_farm_harvest,
            prog._hard_store_restock,
            prog._hard_bus_trip_budget,
            prog._logic_assignment,
            prog._logic_two_attribute,
        ],
    }

    def _unique_from(fns: list, seed: int, max_tries: int = 80):
        rng = random.Random(seed)
        for t in range(max_tries):
            fn = rng.choice(fns)
            try:
                sample = fn(random.Random(seed * 10007 + t * 17 + rng.randint(0, 10_000)))
            except Exception:
                continue
            if not (sample.meta_extra or {}).get("template_family"):
                sample = prog._with_family(sample, getattr(fn, "__name__", "family"))
            key = re.sub(r"\s+", " ", sample.prompt.strip())
            if key in seen:
                continue
            seen.add(key)
            return sample
        return None

    all_rows: list[dict] = []
    tries = 0
    max_tries = 80000
    gen_buffer = 25  # extra per band for coldstart dedup losses at compose
    log(f"gen_start pass8_by_band={dict(pass8_by_band)} quotas={BAND_QUOTAS}")

    for band, quota in BAND_QUOTAS.items():
        need = max(0, quota - pass8_by_band.get(band, 0)) + gen_buffer
        fns = band_fns[band]
        got = 0
        while got < need and tries < max_tries:
            tries += 1
            seed = 12000 + tries + hash(band) % 1000
            sample = _unique_from(fns, seed)
            if sample is None:
                continue
            row = sample_to_row(sample, seed=seed)
            if not isinstance(row, dict):
                continue
            # Nudge metadata so heuristic band matches target bucket
            md = dict(row.get("metadata") or {})
            if band == "easy":
                md["num_steps"] = min(int(md.get("num_steps") or 2), 2)
                md["difficulty_tag"] = "easy"
            elif band == "medium":
                md["num_steps"] = min(max(int(md.get("num_steps") or 3), 3), 4)
                md["difficulty_tag"] = "medium"
            elif band == "hard":
                md["num_steps"] = max(int(md.get("num_steps") or 5), 5)
                md["difficulty_tag"] = "hard"
            else:  # deferred
                md["num_steps"] = max(int(md.get("num_steps") or 7), 7)
                md["difficulty_tag"] = "hard"
            row["metadata"] = md
            row = stamp_compose_band(row, band)
            all_rows.append(row)
            got += 1
            if got % 100 == 0 or got == need:
                log(f"gen band={band} {got}/{need} tries={tries}")

    write_jsonl(out, all_rows)
    log(f"gen_done total={len(all_rows)} tries={tries} by_band={Counter(compose_band(r) for r in all_rows)}")
    return out


def filter_by_band(pool: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {b: [] for b in BAND_QUOTAS}
    for row in read_jsonl(pool):
        buckets[compose_band(row)].append(row)
    log(f"filter_by_band " + ", ".join(f"{b}={len(buckets[b])}" for b in BAND_QUOTAS))
    return buckets


def maybe_rqa(rows: list[dict], label: str) -> list[dict]:
    if not rows or SKIP_RQA:
        log(f"skip RQA for {label} (SKIP_RQA={SKIP_RQA})")
        return rows
    key = ensure_api_key()
    if not key:
        log(f"skip RQA for {label} (no DEEPSEEK_API_KEY)")
        return rows
    inp = WORK / f"{label}_pre_rqa.jsonl"
    out = WORK / f"{label}_rqa.jsonl"
    write_jsonl(inp, rows)
    try:
        run_py(
            [
                str(DATA1 / "scripts" / "apply_full_reverse_qa_sft.py"),
                "--in",
                str(inp),
                "--out",
                str(out),
                "--workers",
                "32",
            ],
            env_extra={
                "REVERSE_QA": "1",
                "REVERSE_QA_MODEL": "deepseek-v4-flash",
                "REVERSE_QA_MODE": "full",
            },
        )
        return read_jsonl(out) if out.is_file() else rows
    except Exception as exc:
        log(f"RQA failed for {label}: {exc}; keeping pre-RQA")
        return rows


def _row_ok(row: dict, cold_keys: set[str], seen: set[str]) -> bool:
    k = prompt_key(row)
    if not k or k in seen or k in cold_keys:
        return False
    if str(row.get("domain") or "").startswith("mcq"):
        return False
    spec = row.get("answer_spec") or {}
    if spec.get("type") not in ("integer", "logic_json", None) and "mcq" in str(
        spec.get("type", "")
    ).lower():
        return False
    return True


def compose(
    stamped: list[dict],
    generated: list[dict],
    old_cand: list[dict],
) -> list[dict]:
    """Fill exactly BAND_QUOTAS from pass@8 empirical + generated + fill."""
    cold_keys = {prompt_key(r) for r in read_jsonl(COLD) if prompt_key(r)}
    seen: set[str] = set()
    out: list[dict] = []
    picked: Counter[str] = Counter()

    # Pool per band: pass@8 empirical first, then generated, then old candidates
    pools: dict[str, list[dict]] = {b: [] for b in BAND_QUOTAS}
    for r in stamped:
        b = band_of(r)
        if b in BAND_QUOTAS:
            pools[b].append(r)
    for r in generated:
        pools[compose_band(r)].append(r)
    for r in old_cand:
        pools[compose_band(r)].append(r)

    for b in BAND_QUOTAS:
        # pass@8 empirical rows rank first within each band
        pools[b].sort(
            key=lambda r: (
                0 if band_of(r) == b else 1,
                -num_steps(r),
                str(r.get("problem_id") or ""),
            )
        )

    for band, quota in BAND_QUOTAS.items():
        n = 0
        for r in pools[band]:
            if n >= quota:
                break
            if not _row_ok(r, cold_keys, seen):
                continue
            seen.add(prompt_key(r))
            row = stamp_compose_band(dict(r), band) if not band_of(r) else dict(r)
            out.append(row)
            n += 1
            picked[band] += 1
        if n < quota:
            log(f"WARNING band={band} shortfall {n}/{quota}")

    log(f"compose_balanced total={len(out)} picked={dict(picked)}")
    return out[:TARGET_TOTAL]


def main() -> int:
    global WORK, PASS8_LOCAL
    if REUSE_WORK:
        WORK = Path(REUSE_WORK)
        PASS8_LOCAL = WORK / "pass8_stamped.jsonl"
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"START work={WORK} target={TARGET_TOTAL} quotas={BAND_QUOTAS} skip_rqa={SKIP_RQA}")

    pool_path = WORK / "gen" / "release_corpora" / "rlvr_train.jsonl"

    if REUSE_WORK and pool_path.is_file() and BALANCED_ONLY:
        stamped = read_jsonl(PASS8_LOCAL)
        if not stamped:
            stamped = pull_pass8_stamped()
        generated = read_jsonl(pool_path)
        log(f"balanced_only pool={len(generated)} stamped={len(stamped)}")
    elif REUSE_WORK and pool_path.is_file():
        stamped = read_jsonl(PASS8_LOCAL)
        if not stamped:
            stamped = pull_pass8_stamped()
        generated = read_jsonl(pool_path)
        log(f"reuse pool={len(generated)} stamped={len(stamped)}")
    else:
        stamped = pull_pass8_stamped()
        pool_path = generate_pool()
        generated = read_jsonl(pool_path)

    old = read_jsonl(CAND)
    if not old or len(old) <= TARGET_TOTAL:
        old = read_jsonl(V5) + old
    log(f"old_cand_pool={len(old)}")

    if not SKIP_RQA and generated:
        by_band = filter_by_band(pool_path)
        enriched: list[dict] = []
        for band, rows in by_band.items():
            enriched.extend(maybe_rqa(rows[:200], band))
        generated = enriched
        write_jsonl(WORK / "generated_rqa.jsonl", generated)

    final = compose(stamped, generated, old)
    if len(final) < TARGET_TOTAL:
        raise RuntimeError(f"could not reach {TARGET_TOTAL}, got {len(final)}")

    # stats
    steps = Counter(num_steps(r) for r in final)
    bands = Counter(compose_band(r) for r in final)
    emp_bands = Counter(band_of(r) or "heuristic" for r in final)
    domains = Counter(str(r.get("domain")) for r in final)
    stats = {
        "total": len(final),
        "band_quotas": BAND_QUOTAS,
        "compose_bands": dict(bands),
        "empirical_pass8_bands": dict(emp_bands),
        "domains": dict(domains),
        "num_steps": dict(sorted(steps.items())),
        "work_dir": str(WORK),
    }
    write_jsonl(CAND, final)
    write_jsonl(V5, final)
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # uniqueness gate
    run_py(
        [
            str(DATA1 / "scripts" / "assert_prompt_uniqueness_v5.py"),
            "--sft",
            str(COLD),
            "--rlvr",
            str(CAND),
        ]
    )
    log(f"DONE {json.dumps(stats)}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FAILED {e}")
        raise
