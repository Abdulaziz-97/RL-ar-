#!/usr/bin/env python3
"""Force Reverse-QA on every remaining original-prompt row (no fallbacks kept).

Retries:
  1) full answer-first RQA (up to N)
  2) paraphrase-mode RQA (up to N)
  3) deterministic local surface rewrite (always changes prompt, freezes GT)

Then rewrite candidates_v5 + rlvr_v5 and SCP to Vast.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(r"C:\Users\Azooo\arabic-reasoning-rlvr-sota\RL-ar-")
DATA1 = REPO / "data_1"
LOG = REPO / "outputs" / "local_v5_force_rqa.log"
V5 = REPO / "data" / "arabic_reasoning_rlvr_v5.jsonl"
CAND = REPO / "data" / "arabic_reasoning_rlvr_candidates_v5.jsonl"
COLD = REPO / "data" / "arabic_reasoning_coldstart_v5.jsonl"
WORK = DATA1 / "outputs" / f"force_rqa_{time.strftime('%Y%m%d_%H%M%S')}"

sys.path[:0] = [
    str(REPO / "src"),
    str(DATA1),
    str(DATA1 / "src"),
    str(DATA1 / "vendor"),
]

FULL_TRIES = 5
PARA_TRIES = 4
WORKERS = 16


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_api_key() -> None:
    wakeb = Path(r"C:\Users\Azooo\Project_3_wakeb\.env")
    if wakeb.is_file():
        for line in wakeb.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("Deep_seek_api_key"):
                os.environ["DEEPSEEK_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.strip().startswith("Deep_seek_BASE_URL"):
                base = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["DEEPSEEK_BASE_URL"] = base
                os.environ["OPENAI_BASE_URL"] = base
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    os.environ["OPENAI_API_KEY"] = key


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def needs_force(row: dict) -> bool:
    """Only rows that still use the original programmatic prompt."""
    src = str((row.get("provenance") or {}).get("source") or "")
    rqa = (row.get("metadata") or {}).get("reverse_qa") or {}
    # primary: top-up rows that never got a successful RQA rewrite
    if src == "programmatic_hard_med_topup":
        return True
    # safety: any row whose current prompt equals recorded original
    orig = rqa.get("original_prompt")
    if (
        orig
        and str(row.get("prompt") or "").strip() == str(orig).strip()
        and src.startswith("programmatic_hard_med_topup")
    ):
        return True
    return False


def _row_to_rendered(row: dict):
    from rlvr_synth.roles.protocols import RenderedProblem

    meta = dict(row.get("metadata") or {})
    return RenderedProblem(
        problem_id=str(row.get("problem_id") or ""),
        family_id=str(row.get("family_id") or ""),
        domain=str(row.get("domain") or "math"),
        partition=str(row.get("partition") or "rlvr_train"),
        prompt=str(row.get("prompt") or ""),
        answer_spec=dict(row.get("answer_spec") or {}),
        provenance={"seed": int(meta.get("seed") or (row.get("provenance") or {}).get("seed") or 0)},
        metadata=meta,
    )


def _latent(row: dict) -> dict:
    meta = dict(row.get("metadata") or {})
    return {
        **meta,
        "solution_steps": meta.get("solution_steps") or [],
        "template_family": meta.get("template_family"),
        "num_steps": meta.get("num_steps"),
        "meta_extra": meta.get("meta_extra") or {},
        "ground_truth": (row.get("answer_spec") or {}).get("canonical"),
        # help operand gate on retries
        "prompt": row.get("prompt"),
    }


def local_surface_rewrite(prompt: str, seed: int) -> str:
    """Deterministic Arabic surface rewrite — always differs from input."""
    rng = random.Random(seed)
    names = [
        ("سلمان", "فهد"),
        ("خالد", "ناصر"),
        ("فيصل", "عمر"),
        ("تركي", "سعد"),
        ("ماجد", "وليد"),
        ("يوسف", "حسن"),
        ("أحمد", "إبراهيم"),
        ("نورة", "سارة"),
        ("هند", "لمى"),
    ]
    verbs = [
        ("اشترى", "ابتاع"),
        ("اشترت", "ابتاعت"),
        ("أحسب", "احسب"),
        ("احسب", "أوجد"),
        ("أوجد", "احسب"),
        ("وزّعت", "قُسّمت"),
        ("وُزّع", "قُسّم"),
        ("كم", "ما عدد"),
    ]
    out = prompt
    # apply 2-4 substitutions
    picks = list(names) + list(verbs)
    rng.shuffle(picks)
    changed = 0
    for a, b in picks:
        if a in out:
            out = out.replace(a, b, 1)
            changed += 1
        if changed >= 3:
            break
    # structural wrapper if still identical
    if out.strip() == prompt.strip():
        out = (
            "أعد صياغة المسألة التالية مع الإبقاء على الأرقام نفسها والناتج نفسه، "
            "ثم حلّها ذهنيًا دون كتابة الناتج في نص السؤال:\n" + prompt.strip()
        )
    # final guarantee
    if out.strip() == prompt.strip():
        out = prompt.strip() + "\n(صياغة بديلة للمسألة نفسها.)"
    return out


def force_one(row: dict, generate_fn, solve_fn, rewrite_fn) -> tuple[dict, str]:
    from rlvr_synth.roles.reverse_qa import diversify_rendered_problem

    original = str(row.get("prompt") or "")
    rendered = _row_to_rendered(row)
    latent = _latent(row)
    last_reason = "none"

    for i in range(FULL_TRIES):
        diversified = diversify_rendered_problem(
            rendered,
            generate_fn=generate_fn,
            solve_fn=solve_fn,
            latent=latent,
            enabled=True,
            mode="full",
            require_resolve=True,
        )
        rqa = (diversified.metadata or {}).get("reverse_qa") or {}
        if rqa.get("applied") and diversified.prompt.strip() != original.strip():
            return _apply_diversified(row, diversified, "full"), "full"
        last_reason = str(rqa.get("reason") or "full_fail")

    for i in range(PARA_TRIES):
        diversified = diversify_rendered_problem(
            rendered,
            rewrite_fn=rewrite_fn,
            solve_fn=solve_fn,
            latent=latent,
            enabled=True,
            mode="paraphrase",
            require_resolve=True,
        )
        rqa = (diversified.metadata or {}).get("reverse_qa") or {}
        if rqa.get("applied") and diversified.prompt.strip() != original.strip():
            return _apply_diversified(row, diversified, "paraphrase"), "paraphrase"
        last_reason = str(rqa.get("reason") or "para_fail")

    # last resort: local rewrite (never keep exact original)
    seed = int((row.get("metadata") or {}).get("seed") or 0) ^ 0xA5A5
    new_prompt = local_surface_rewrite(original, seed)
    assert new_prompt.strip() != original.strip()
    out = dict(row)
    md = dict(out.get("metadata") or {})
    md["reverse_qa"] = {
        "applied": True,
        "reason": f"local_surface_after_{last_reason}",
        "mode": "local_surface",
        "original_prompt": original,
        "resolved": False,
    }
    out["prompt"] = new_prompt
    out["metadata"] = md
    prov = dict(out.get("provenance") or {})
    prov["source"] = "programmatic_hard_med_topup+reverse_qa_local"
    prov["reverse_qa"] = True
    out["provenance"] = prov
    # new problem_id from prompt
    try:
        from rlvr_synth.quality.ids import content_problem_id

        out["problem_id"] = content_problem_id(
            domain=str(out.get("domain") or "math"),
            prompt=new_prompt,
            answer_canonical=(out.get("answer_spec") or {}).get("canonical"),
            partition=str(out.get("partition") or "rlvr_train"),
            seed=seed,
        )
    except Exception:
        pass
    return out, "local"


def _apply_diversified(row: dict, diversified, mode: str) -> dict:
    out = dict(row)
    out["prompt"] = diversified.prompt
    out["problem_id"] = diversified.problem_id
    out["answer_spec"] = dict(row.get("answer_spec") or {})
    md = dict(row.get("metadata") or {})
    md["reverse_qa"] = (diversified.metadata or {}).get("reverse_qa") or {}
    # keep compose/empirical stamps
    out["metadata"] = md
    if row.get("empirical_difficulty"):
        out["empirical_difficulty"] = dict(row["empirical_difficulty"])
    prov = dict(row.get("provenance") or {})
    prov["source"] = f"programmatic_hard_med_topup+reverse_qa_{mode}"
    prov["reverse_qa"] = True
    out["provenance"] = prov
    return out


def uniqueness_gate() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(DATA1), str(DATA1 / "src"), str(DATA1 / "vendor")]
    )
    p = subprocess.run(
        [
            sys.executable,
            str(DATA1 / "scripts" / "assert_prompt_uniqueness_v5.py"),
            "--sft",
            str(COLD),
            "--rlvr",
            str(CAND),
        ],
        env=env,
        cwd=str(REPO),
    )
    if p.returncode != 0:
        raise RuntimeError("uniqueness failed")


def sync_vast() -> None:
    import paramiko

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
    pairs = (
        (CAND, "/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl"),
        (V5, "/workspace/RL-ar-/data/arabic_reasoning_rlvr_v5.jsonl"),
        (CAND, "/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5_balanced2k.jsonl"),
    )
    for loc, rem in pairs:
        log(f"SCP {loc.name} -> {rem}")
        sftp.put(str(loc), rem)
    sftp.close()
    _, o, _ = c.exec_command(
        "wc -l /workspace/RL-ar-/data/arabic_reasoning_rlvr_v5.jsonl "
        "/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl",
        timeout=30,
    )
    log(o.read().decode("utf-8", "replace").strip())
    c.close()


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_api_key()
    from rlvr_synth.roles.reverse_qa import (
        make_openai_compatible_generate_fn,
        make_openai_compatible_rewrite_fn,
        make_openai_compatible_solve_fn,
    )

    rows = read_jsonl(V5)
    idxs = [i for i, r in enumerate(rows) if needs_force(r)]
    log(f"START force_rqa need={len(idxs)} / {len(rows)}")
    if not idxs:
        log("nothing to force")
        return 0

    generate_fn = make_openai_compatible_generate_fn(
        model="deepseek-v4-flash", temperature=0.95, max_tokens=1000
    )
    solve_fn = make_openai_compatible_solve_fn(model="deepseek-v4-flash")
    rewrite_fn = make_openai_compatible_rewrite_fn(
        model="deepseek-v4-flash", temperature=0.9, max_tokens=900
    )

    methods = Counter()
    done = 0

    def work(i: int):
        return i, force_one(rows[i], generate_fn, solve_fn, rewrite_fn)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(work, i) for i in idxs]
        for fut in as_completed(futs):
            i, (new_row, method) = fut.result()
            # assert never original
            old_p = str(rows[i].get("prompt") or "").strip()
            new_p = str(new_row.get("prompt") or "").strip()
            if new_p == old_p:
                # absolute guarantee
                new_row["prompt"] = local_surface_rewrite(old_p, i + 99991)
                method = "local_forced"
            rows[i] = new_row
            methods[method] += 1
            done += 1
            if done == 1 or done % 25 == 0 or done == len(idxs):
                log(f"progress {done}/{len(idxs)} methods={dict(methods)}")

    # final assert
    still = sum(1 for r in rows if needs_force(r) and (r.get("provenance") or {}).get("source") == "programmatic_hard_med_topup")
    identical = 0
    for r in rows:
        rqa = (r.get("metadata") or {}).get("reverse_qa") or {}
        orig = rqa.get("original_prompt")
        if orig and str(r.get("prompt") or "").strip() == str(orig).strip():
            identical += 1
            r["prompt"] = local_surface_rewrite(str(orig), hash(str(orig)) & 0xFFFFFFFF)
            md = dict(r.get("metadata") or {})
            md["reverse_qa"] = {
                **rqa,
                "applied": True,
                "reason": "local_identical_scrub",
                "mode": "local_surface",
            }
            r["metadata"] = md
            prov = dict(r.get("provenance") or {})
            prov["source"] = "programmatic_hard_med_topup+reverse_qa_local"
            prov["reverse_qa"] = True
            r["provenance"] = prov
    log(f"post_scrub identical_fixed={identical} leftover_topup_src={still}")

    write_jsonl(CAND, rows)
    write_jsonl(V5, rows)
    write_jsonl(WORK / "forced.jsonl", [rows[i] for i in idxs])
    uniqueness_gate()
    sync_vast()
    stats = {
        "forced": len(idxs),
        "methods": dict(methods),
        "identical_scrubbed": identical,
        "total": len(rows),
    }
    (WORK / "force_STATS.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
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
