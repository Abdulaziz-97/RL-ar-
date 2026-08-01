#!/usr/bin/env python3
"""Apply Flash Reverse-QA to new programmatic rows, sync to Vast, arm pass@8.

Flow:
  1) RQA only rows with provenance programmatic_hard_med_topup (keep pass@8 leftovers as-is).
  2) Merge → write candidates_v5 + rlvr_v5 (2000 balanced).
  3) SCP both files to Vast.
  4) Deploy waiter: when GPUs free (SFT bench done), launch 2-shard HF pass@8
     on the new candidates with the same SFT checkpoint.
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
LOG = REPO / "outputs" / "local_v5_rqa_pass8.log"
V5 = REPO / "data" / "arabic_reasoning_rlvr_v5.jsonl"
CAND = REPO / "data" / "arabic_reasoning_rlvr_candidates_v5.jsonl"
COLD = REPO / "data" / "arabic_reasoning_coldstart_v5.jsonl"
WORK = DATA1 / "outputs" / f"rlvr_rqa_pass8_{time.strftime('%Y%m%d_%H%M%S')}"
STATS = REPO / "data" / "arabic_reasoning_rlvr_v5_rqa_STATS.json"

HOST = "8.243.214.78"
PORT = 49198
SSH_PASS = "aziz"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    wakeb = Path(r"C:\Users\Azooo\Project_3_wakeb\.env")
    for line in wakeb.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("Deep_seek_api_key"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["DEEPSEEK_API_KEY"] = key
            # also map base URL if present
            break
    for line in wakeb.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("Deep_seek_BASE_URL"):
            os.environ.setdefault(
                "DEEPSEEK_BASE_URL",
                line.split("=", 1)[1].strip().strip('"').strip("'"),
            )
            os.environ.setdefault(
                "OPENAI_BASE_URL",
                os.environ["DEEPSEEK_BASE_URL"],
            )
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    os.environ.setdefault("OPENAI_API_KEY", key)
    return key


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def band_of(row: dict) -> str:
    b = str((row.get("empirical_difficulty") or {}).get("band") or "")
    return "hard" if b == "hard_diagnostic" else b


def run_rqa(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    ensure_api_key()
    inp = WORK / "need_rqa.jsonl"
    out = WORK / "need_rqa_out.jsonl"
    write_jsonl(inp, rows)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(DATA1), str(DATA1 / "src"), str(DATA1 / "vendor")]
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["REVERSE_QA"] = "1"
    env["REVERSE_QA_MODEL"] = "deepseek-v4-flash"
    env["REVERSE_QA_MODE"] = "full"
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or env["DEEPSEEK_API_KEY"]
    if env.get("DEEPSEEK_BASE_URL"):
        env.setdefault("OPENAI_BASE_URL", env["DEEPSEEK_BASE_URL"])
    cmd = [
        sys.executable,
        str(DATA1 / "scripts" / "apply_full_reverse_qa_sft.py"),
        "--in",
        str(inp),
        "--out",
        str(out),
        "--workers",
        "32",
        "--model",
        "deepseek-v4-flash",
    ]
    log("RUN RQA " + " ".join(cmd))
    p = subprocess.run(cmd, env=env, cwd=str(REPO))
    if p.returncode != 0:
        raise RuntimeError(f"RQA failed exit={p.returncode}")
    return read_jsonl(out)


def ssh_client():
    import paramiko

    key = paramiko.Ed25519Key.from_private_key_file(
        str(Path.home() / ".ssh" / "id_ed25519"), password=SSH_PASS
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST,
        port=PORT,
        username="root",
        pkey=key,
        timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )
    return c


def sync_to_vast(local_cand: Path, local_v5: Path) -> None:
    rem_cand = "/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl"
    rem_v5 = "/workspace/RL-ar-/data/arabic_reasoning_rlvr_v5.jsonl"
    rem_cand_new = "/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5_balanced2k.jsonl"
    c = ssh_client()
    sftp = c.open_sftp()
    for loc, rem in (
        (local_cand, rem_cand),
        (local_v5, rem_v5),
        (local_cand, rem_cand_new),
    ):
        log(f"SCP {loc.name} -> {rem}")
        sftp.put(str(loc), rem)
    sftp.close()
    _, o, _ = c.exec_command(
        f"wc -l {rem_cand} {rem_v5} {rem_cand_new}", timeout=30
    )
    log(o.read().decode("utf-8", "replace").strip())
    c.close()


PASS8_WAITER = r'''#!/usr/bin/env python3
"""Wait for GPUs free, then launch 2-shard HF pass@8 on balanced 2k candidates."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

STATUS = Path("/workspace/outputs/pass8_balanced2k_STATUS.json")
LOG = Path("/workspace/outputs/pass8_balanced2k_waiter.log")
CAND = Path("/workspace/RL-ar-/data/arabic_reasoning_rlvr_candidates_v5.jsonl")
SFT = Path("/workspace/outputs/sft_coldstart_v4_v5_20260801_174829")
OUT = Path("/workspace/RL-ar-/data_1/outputs/v4_regen/v5_20260801_174829/pass8_balanced2k")
PY = "/workspace/RL-ar-/.venv/bin/python"
if not Path(PY).exists():
    PY = "/venv/main/bin/python"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kw):
    STATUS.write_text(json.dumps({"ts": time.time(), **kw}, indent=2), encoding="utf-8")


def pgrep(pat: str) -> str:
    r = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True)
    return "\n".join(
        ln for ln in (r.stdout or "").splitlines() if "pgrep" not in ln and ln.strip()
    )


def gpu_busy() -> bool:
    # block if generative SFT bench or existing pass8/grpo train holds GPUs
    for pat in (
        "run_araeval_generative",
        "calibrate_v4_pass8",
        "trl.*grpo",
        "train_grpo",
        "accelerate launch",
    ):
        if pgrep(pat).strip():
            return True
    return False


def launch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # clear previous balanced2k shards only (keep old pass8 archive)
    for p in OUT.glob("cand_shard*.jsonl"):
        p.unlink(missing_ok=True)
    for p in OUT.glob("cal_shard*.json"):
        p.unlink(missing_ok=True)

    n = sum(1 for _ in CAND.open(encoding="utf-8") if _.strip())
    log(f"launch pass@8 on {CAND} n={n} sft={SFT}")
    write_status(state="launching", n=n, out=str(OUT))

    env_base = os.environ.copy()
    env_base["HF_HOME"] = env_base.get("HF_HOME", "/workspace/hf")
    env_base["TRANSFORMERS_CACHE"] = env_base.get("TRANSFORMERS_CACHE", "/workspace/hf")
    env_base["TOKENIZERS_PARALLELISM"] = "false"
    env_base["PYTHONUNBUFFERED"] = "1"

    batch = "4"  # stable on 2x5090 with T06-merge
    for shard in (0, 1):
        log_path = Path(f"/workspace/outputs/pass8_balanced2k_shard{shard}.log")
        cmd = [
            PY,
            "/workspace/RL-ar-/data_1/scripts/calibrate_v4_pass8.py",
            "--candidates",
            str(CAND),
            "--sft-checkpoint",
            str(SFT),
            "--out-calibration",
            str(OUT / f"cal_shard{shard}.json"),
            "--out-candidates",
            str(OUT / f"cand_shard{shard}.jsonl"),
            "--shard-id",
            str(shard),
            "--num-shards",
            "2",
            "--backend",
            "hf",
            "--n",
            "8",
            "--temperature",
            "0.7",
            "--max-new-tokens",
            "768",
            "--prompt-batch-size",
            batch,
            "--merge-sft-lora",
        ]
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        with log_path.open("w") as lf:
            subprocess.Popen(
                cmd,
                cwd="/workspace/RL-ar-",
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log(f"launched shard={shard} log={log_path}")
    write_status(state="running", n=n, out=str(OUT), batch=batch)
    log("pass@8 launched both shards")


def main() -> int:
    log("waiter start")
    write_status(state="waiting_gpus")
    # do not kill SFT bench — wait for it
    while gpu_busy():
        busy = []
        for pat in ("run_araeval_generative", "calibrate_v4_pass8", "trl", "accelerate"):
            if pgrep(pat).strip():
                busy.append(pat)
        log(f"gpus busy ({','.join(busy) or 'unknown'}); sleep 60")
        write_status(state="waiting_gpus", busy=busy)
        time.sleep(60)

    if not CAND.is_file():
        write_status(state="error", error="missing candidates")
        raise SystemExit("missing candidates")
    if not (SFT / "adapter_config.json").is_file():
        write_status(state="error", error="missing sft")
        raise SystemExit("missing sft")

    launch()
    # stay alive until shards finish
    while True:
        procs = pgrep("calibrate_v4_pass8")
        s0 = OUT / "cand_shard0.jsonl"
        s1 = OUT / "cand_shard1.jsonl"
        n0 = sum(1 for _ in s0.open(encoding="utf-8") if _.strip()) if s0.is_file() else 0
        n1 = sum(1 for _ in s1.open(encoding="utf-8") if _.strip()) if s1.is_file() else 0
        write_status(
            state="running" if procs.strip() else "done",
            shard0=n0,
            shard1=n1,
            procs=bool(procs.strip()),
        )
        if not procs.strip() and n0 > 0 and n1 > 0:
            log(f"DONE shards {n0}+{n1}")
            write_status(state="done", shard0=n0, shard1=n1)
            return 0
        if not procs.strip() and (n0 == 0 or n1 == 0):
            log(f"pass8 exited early shard0={n0} shard1={n1}")
            write_status(state="error", shard0=n0, shard1=n1, error="early_exit")
            return 1
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def deploy_pass8_waiter() -> None:
    c = ssh_client()
    sftp = c.open_sftp()
    remote = "/workspace/outputs/vast_pass8_balanced2k_waiter.py"
    with sftp.file(remote, "w") as f:
        f.write(PASS8_WAITER)
    sftp.chmod(remote, 0o755)
    sftp.close()

    # kill previous balanced2k waiter if any; do NOT kill SFT bench or GRPO waiter yet
    cmd = (
        "pkill -f vast_pass8_balanced2k_waiter.py 2>/dev/null || true; "
        "sleep 1; "
        "nohup /venv/main/bin/python -u /workspace/outputs/vast_pass8_balanced2k_waiter.py "
        ">/workspace/outputs/pass8_balanced2k_waiter.nohup 2>&1 & "
        "echo launched_pid=$!; "
        "sleep 2; "
        "pgrep -af vast_pass8_balanced2k_waiter | grep -v pgrep; "
        "head -n 5 /workspace/outputs/pass8_balanced2k_waiter.log 2>/dev/null || true"
    )
    _, o, e = c.exec_command(cmd, timeout=60)
    log(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        log("stderr " + err[:500])
    c.close()


def uniqueness_gate() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO / "src"),
            str(DATA1),
            str(DATA1 / "src"),
            str(DATA1 / "vendor"),
        ]
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


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"START work={WORK}")
    rows = read_jsonl(V5)
    if len(rows) != 2000:
        log(f"WARNING expected 2000 got {len(rows)}")

    need = [
        r
        for r in rows
        if (r.get("provenance") or {}).get("source") == "programmatic_hard_med_topup"
    ]
    keep = [
        r
        for r in rows
        if (r.get("provenance") or {}).get("source") != "programmatic_hard_med_topup"
    ]
    log(f"keep={len(keep)} need_rqa={len(need)} bands_before={Counter(band_of(r) for r in rows)}")

    rqa_rows = run_rqa(need)
    if len(rqa_rows) != len(need):
        raise RuntimeError(f"RQA length mismatch {len(need)} -> {len(rqa_rows)}")
    # preserve compose/empirical band stamps (RQA may change problem_id/prompt)
    merged_new = []
    applied = 0
    for orig, r in zip(need, rqa_rows):
        md = dict(r.get("metadata") or {})
        rqa = md.get("reverse_qa") or {}
        if rqa.get("applied"):
            applied += 1
            prov = dict(r.get("provenance") or {})
            prov["source"] = "programmatic_hard_med_topup+reverse_qa_full"
            prov["reverse_qa"] = True
            r["provenance"] = prov
        # always keep intended band from pre-RQA row
        if orig.get("empirical_difficulty"):
            r["empirical_difficulty"] = dict(orig["empirical_difficulty"])
        omd = orig.get("metadata") or {}
        if omd.get("compose_band") or omd.get("difficulty_tag"):
            md["compose_band"] = omd.get("compose_band") or omd.get("difficulty_tag")
            md["difficulty_tag"] = omd.get("difficulty_tag") or omd.get("compose_band")
            md["num_steps"] = omd.get("num_steps", md.get("num_steps"))
            r["metadata"] = md
        merged_new.append(r)

    final = keep + merged_new
    # rebalance not needed if we kept all; assert length
    if len(final) != len(rows):
        log(f"WARNING length drift {len(rows)} -> {len(final)}")

    bands = Counter(band_of(r) for r in final)
    write_jsonl(CAND, final)
    write_jsonl(V5, final)
    stats = {
        "total": len(final),
        "rqa_input": len(need),
        "rqa_applied": applied,
        "rqa_fallback": len(need) - applied,
        "bands": dict(bands),
        "work_dir": str(WORK),
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    uniqueness_gate()
    log(f"local_done {json.dumps(stats)}")

    sync_to_vast(CAND, V5)
    deploy_pass8_waiter()
    log("DONE rqa+sync+pass8_waiter_armed")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FAILED {e}")
        raise
