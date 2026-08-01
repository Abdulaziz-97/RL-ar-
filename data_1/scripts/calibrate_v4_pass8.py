#!/usr/bin/env python3
"""Strict pass@8 calibration on the fresh SFT lineage for RLVR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_contracts.verifiers import verify_answer
from rlvr_synth.calibration.pass_at_n import (
    assign_band,
    calibrate_pass_at_n,
    write_calibration,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_verify_fn():
    def _verify(prob: dict[str, Any], completion: str) -> bool:
        return bool(
            verify_answer(
                completion, prob.get("answer_spec") or {}, from_completion=True
            ).ok
        )

    return _verify


def _make_hf_generate_fn(
    *,
    sft_checkpoint: Path,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Return (generate_n_fn, checkpoint_hash, tokenizer_id).

    ``generate_n_fn(prompt, seeds)`` draws len(seeds) samples in one
    ``num_return_sequences`` forward (same N/temp/top_p/max tokens). Parent RNG
    seed is ``seeds[0]`` so re-runs are reproducible without 8 serial generates.
    """
    from rlvr_pipeline.lineage import load_lineage_model, read_lineage
    import torch

    meta = read_lineage(sft_checkpoint)
    base = (meta.base_model if meta else None) or "Qwen/Qwen3.5-4B"
    instruction = meta.instruction_adapter if meta else None
    model, tokenizer = load_lineage_model(
        base_model=base,
        instruction_adapter=instruction,
        trainable_adapter=str(sft_checkpoint),
        merge_instruction=True,
        is_trainable=False,
        device_map="auto",
    )
    model.eval()
    ckpt_hash = _file_hash(sft_checkpoint / "adapter_config.json") if (
        sft_checkpoint / "adapter_config.json"
    ).exists() else _file_hash(sft_checkpoint)

    def _gen_n(prompt: str, seeds: list[int]) -> list[str]:
        if not seeds:
            return []
        torch.manual_seed(int(seeds[0]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seeds[0]))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=len(seeds),
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]

    return _gen_n, ckpt_hash, getattr(tokenizer, "name_or_path", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-calibration", type=Path, required=True)
    parser.add_argument("--out-candidates", type=Path, required=True,
                        help="Candidates stamped with empirical_difficulty")
    parser.add_argument("--sft-checkpoint", type=Path, default=None)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stub", action="store_true",
                        help="Credential/GPU-free stub generator for tests")
    args = parser.parse_args()

    rows = _read_jsonl(args.candidates)
    if args.num_shards < 1:
        raise SystemExit("num_shards must be >= 1")
    shard_rows = [
        r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id
    ]

    system_prompt = args.system_prompt
    if args.stub:
        from rlvr_synth.calibration.pass_at_n import stub_generate_fn_factory

        generate_fn = stub_generate_fn_factory()
        generate_n_fn = None
        checkpoint_hash = "stub"
        tokenizer_id = "stub"
    else:
        if args.sft_checkpoint is None:
            raise SystemExit("--sft-checkpoint is required unless --stub")
        generate_n_fn, checkpoint_hash, tokenizer_id = _make_hf_generate_fn(
            sft_checkpoint=args.sft_checkpoint,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generate_fn = None

    results = calibrate_pass_at_n(
        shard_rows,
        generate_fn=generate_fn,
        generate_n_fn=generate_n_fn,
        verify_fn=_make_verify_fn(),
        n=args.n,
        base_seed=args.base_seed,
        checkpoint_hash=checkpoint_hash,
        tokenizer_id=tokenizer_id,
        system_prompt=system_prompt,
    )

    # Fail closed on missing/duplicate seeds.
    seen = set()
    for r in results:
        if len(r.seeds) != args.n:
            raise SystemExit(f"bad seed count for {r.problem_id}")
        if len(set(r.seeds)) != args.n:
            raise SystemExit(f"duplicate seeds for {r.problem_id}")
        key = (r.problem_id, tuple(r.seeds))
        if key in seen:
            raise SystemExit(f"duplicate calibration key {key}")
        seen.add(key)

    write_calibration(results, args.out_calibration)

    by_pid = {r.problem_id: r for r in results}
    stamped = []
    for row in shard_rows:
        pid = str(row.get("problem_id") or row.get("id") or "")
        cal = by_pid.get(pid)
        out = dict(row)
        if cal is None:
            raise SystemExit(f"missing calibration for {pid}")
        band = assign_band(cal.pass_fraction)
        emp = {
            "n": cal.n,
            "passes": cal.passes,
            "pass_fraction": cal.pass_fraction,
            "band": band,
            "seeds": cal.seeds,
            "checkpoint_hash": cal.checkpoint_hash,
            "tokenizer_id": cal.tokenizer_id,
            "system_prompt_hash": cal.system_prompt_hash,
            "raw": cal.raw,
        }
        out["empirical_difficulty"] = emp
        out["difficulty_tag"] = band
        meta = dict(out.get("metadata") or {})
        meta["pass_at_8"] = {
            "n": cal.n,
            "passes": cal.passes,
            "pass_fraction": cal.pass_fraction,
        }
        meta["difficulty_band"] = band
        meta["difficulty_source"] = "sft_pass_at_8"
        out["metadata"] = meta
        stamped.append(out)

    _write_jsonl(args.out_candidates, stamped)
    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_candidates": len(shard_rows),
        "bands": {
            b: sum(1 for r in results if r.band == b)
            for b in ("mastered", "easy", "medium", "hard", "deferred")
        },
        "checkpoint_hash": checkpoint_hash,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
