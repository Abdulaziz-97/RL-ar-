#!/usr/bin/env python3
"""Strict pass@8 calibration on the fresh SFT lineage for RLVR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

PACK_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACK_ROOT.parent
sys.path[:0] = [str(PACK_ROOT), str(PACK_ROOT / "src"), str(PACK_ROOT / "vendor"), str(ROOT / "src")]

from rlvr_contracts.verifiers import verify_answer
from rlvr_pipeline.config import DEFAULT_SYSTEM_PROMPT
from rlvr_synth.calibration.pass_at_n import (
    PassAtNResult,
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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_verify_fn():
    def _verify(prob: dict[str, Any], completion: str) -> bool:
        return bool(
            verify_answer(
                completion, prob.get("answer_spec") or {}, from_completion=True
            ).ok
        )

    return _verify


def _apply_chat(tokenizer, system_prompt: str, prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3.5: keep thinking OFF (system prompt already asks for <think> tags).
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_hf_lineage_for_infer(
    *,
    sft_checkpoint: Path,
    merge_sft_lora: bool = True,
):
    """Load T06-merge + SFT LoRA; optionally merge SFT for faster generate (same weights)."""
    from rlvr_pipeline.lineage import load_lineage_model, read_lineage
    import torch

    meta = read_lineage(sft_checkpoint)
    base = (meta.base_model if meta else None) or "Qwen/Qwen3.5-4B"
    instruction = meta.instruction_adapter if meta else None
    print(
        f"[pass8-hf] lineage base={base} instruction={instruction} "
        f"sft={sft_checkpoint} merge_sft_lora={merge_sft_lora}",
        flush=True,
    )
    model, tokenizer = load_lineage_model(
        base_model=base,
        instruction_adapter=instruction,
        trainable_adapter=str(sft_checkpoint),
        merge_instruction=True,
        is_trainable=False,
        device_map="auto",
    )
    model.eval()
    if merge_sft_lora and hasattr(model, "merge_and_unload"):
        print("[pass8-hf] merging SFT LoRA into base for throughput", flush=True)
        model = model.merge_and_unload()
        model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ckpt_hash = _file_hash(sft_checkpoint / "adapter_config.json") if (
        sft_checkpoint / "adapter_config.json"
    ).exists() else _file_hash(sft_checkpoint)
    return model, tokenizer, ckpt_hash, base, instruction


def _make_hf_generate_fn(
    *,
    sft_checkpoint: Path,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Return (generate_n_fn, checkpoint_hash, tokenizer_id) — one prompt at a time."""
    import torch

    model, tokenizer, ckpt_hash, _base, _instr = _load_hf_lineage_for_infer(
        sft_checkpoint=sft_checkpoint, merge_sft_lora=True
    )
    device = next(model.parameters()).device

    def _gen_n(prompt: str, seeds: list[int]) -> list[str]:
        if not seeds:
            return []
        torch.manual_seed(int(seeds[0]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seeds[0]))
        text = _apply_chat(tokenizer, system_prompt, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=len(seeds),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]

    return _gen_n, ckpt_hash, getattr(tokenizer, "name_or_path", "")


def _pid(row: dict[str, Any], fallback: int | str = "") -> str:
    return str(row.get("problem_id") or row.get("id") or fallback)


def _load_resumed_stamped(path: Path) -> dict[str, dict[str, Any]]:
    """Map problem_id → stamped candidate row already flushed to disk."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[_pid(row)] = row
    return out


def _result_from_stamped(row: dict[str, Any]) -> PassAtNResult:
    emp = row.get("empirical_difficulty") or {}
    return PassAtNResult(
        problem_id=_pid(row),
        n=int(emp.get("n") or 0),
        passes=int(emp.get("passes") or 0),
        pass_fraction=float(emp.get("pass_fraction") or 0.0),
        band=str(emp.get("band") or assign_band(float(emp.get("pass_fraction") or 0.0))),
        seeds=list(emp.get("seeds") or []),
        checkpoint_hash=str(emp.get("checkpoint_hash") or ""),
        tokenizer_id=str(emp.get("tokenizer_id") or ""),
        system_prompt_hash=str(emp.get("system_prompt_hash") or ""),
        raw=list(emp.get("raw") or []),
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def _calibrate_hf_batched(
    shard_rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    verify_fn: Callable[[dict[str, Any], str], bool],
    n: int,
    base_seed: int,
    checkpoint_hash: str,
    tokenizer_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_batch_size: int,
    index_offset: int = 0,
    total_shard: int | None = None,
    on_batch: Callable[[list[PassAtNResult], list[dict[str, Any]]], None] | None = None,
) -> list[PassAtNResult]:
    """HF generate over shard_rows; optional incremental flush via on_batch.

    ``index_offset`` / original positions: seeds use absolute shard index
    ``index_offset + local_i`` so resume keeps the seed contract.
    """
    import torch

    device = next(model.parameters()).device
    sys_hash = _hash_text(system_prompt) if system_prompt else ""
    # Keep (absolute_index, row) for correct seeds under resume.
    work = list(enumerate(shard_rows))
    rendered = {
        abs_i: _apply_chat(tokenizer, system_prompt, str(prob.get("prompt", "")))
        for abs_i, prob in ((index_offset + li, r) for li, r in work)
    }
    # Rebuild work with absolute indices
    abs_work = [(index_offset + li, r) for li, r in work]
    results: list[PassAtNResult] = []
    t0 = time.time()
    total = int(total_shard if total_shard is not None else len(shard_rows))
    already = total - len(abs_work)
    done = already
    batch = max(1, int(prompt_batch_size))
    print(
        f"[pass8-hf] generate batch_prompts={batch} (=>{batch * n} seqs/step) "
        f"device={device} resume_skip={already} remaining={len(abs_work)}",
        flush=True,
    )

    for start in range(0, len(abs_work), batch):
        chunk = abs_work[start : start + batch]
        chunk_probs = [r for _, r in chunk]
        chunk_texts = [rendered[abs_i] for abs_i, _ in chunk]
        bsz = len(chunk_texts)
        first_abs = chunk[0][0]
        inputs = tokenizer(
            chunk_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        torch.manual_seed(int(base_seed + first_abs * n))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(base_seed + first_abs * n))
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=int(n),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        batch_results: list[PassAtNResult] = []
        for j, (abs_i, prob) in enumerate(chunk):
            seeds = [base_seed + abs_i * n + k for k in range(n)]
            completions = [
                tokenizer.decode(
                    out[j * n + k][prompt_len:], skip_special_tokens=True
                )
                for k in range(n)
            ]
            flags = [bool(verify_fn(prob, c)) for c in completions]
            passes = sum(flags)
            frac = passes / n if n else 0.0
            batch_results.append(
                PassAtNResult(
                    problem_id=_pid(prob, abs_i),
                    n=n,
                    passes=passes,
                    pass_fraction=frac,
                    band=assign_band(frac),
                    seeds=seeds,
                    checkpoint_hash=checkpoint_hash,
                    tokenizer_id=tokenizer_id,
                    system_prompt_hash=sys_hash,
                    raw=flags,
                )
            )
        results.extend(batch_results)
        if on_batch is not None:
            on_batch(batch_results, chunk_probs)
        done += bsz
        elapsed = max(time.time() - t0, 1e-6)
        # Rate over newly processed only (fairer ETA after resume).
        new_done = done - already
        rate = new_done / elapsed if new_done else 0.0
        eta_s = (total - done) / rate if rate > 0 else float("inf")
        mem = (
            torch.cuda.memory_allocated(device) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        print(
            f"[pass8-hf] {done}/{total} prompts  {rate:.2f} prompts/s  "
            f"{rate * n:.1f} samples/s  ETA {eta_s / 60:.1f} min  "
            f"cuda_alloc={mem:.1f}GiB",
            flush=True,
        )
    return results


def _merge_instruction_base_to_dir(
    *,
    base_model: str,
    instruction_adapter: str,
    out_dir: Path,
) -> Path:
    """Merge T06 (or other instruction LoRA) into base weights for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    marker = out_dir / ".t06_merged_ok"
    if marker.is_file() and (out_dir / "config.json").is_file():
        print(f"[pass8-vllm] reusing merged instruction base: {out_dir}", flush=True)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[pass8-vllm] merging instruction adapter {instruction_adapter} "
        f"into {base_model} → {out_dir}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft = PeftModel.from_pretrained(model, instruction_adapter)
    model = peft.merge_and_unload()
    del peft
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    marker.write_text(
        json.dumps(
            {
                "base_model": base_model,
                "instruction_adapter": instruction_adapter,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def _make_vllm_engine(
    *,
    sft_checkpoint: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_lora_rank: int,
):
    """Build vLLM engine + LoRA request for SFT adapter (1 GPU / process).

    If lineage has instruction_adapter (T06), merge it into a local base dir first.
    Never attach SFT LoRA on raw Qwen when deltas were trained on T06-merged base.
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from rlvr_pipeline.lineage import read_lineage

    meta = read_lineage(sft_checkpoint)
    base = (meta.base_model if meta else None) or "Qwen/Qwen3.5-4B"
    instruction = (meta.instruction_adapter if meta else None) or None
    if instruction:
        instruction = str(instruction).strip() or None
    has_adapter = (sft_checkpoint / "adapter_config.json").exists() or (
        sft_checkpoint / "adapter_model.safetensors"
    ).exists()
    if not has_adapter:
        raise SystemExit(f"SFT adapter missing under {sft_checkpoint}")

    ckpt_hash = _file_hash(sft_checkpoint / "adapter_config.json") if (
        sft_checkpoint / "adapter_config.json"
    ).exists() else _file_hash(sft_checkpoint)

    vllm_base = base
    if instruction:
        # Fail-closed unless we successfully materialize T06-merged weights.
        merge_dir = Path(sft_checkpoint) / "_merged_instruction_base_for_vllm"
        try:
            vllm_base = str(
                _merge_instruction_base_to_dir(
                    base_model=base,
                    instruction_adapter=instruction,
                    out_dir=merge_dir,
                )
            )
        except Exception as exc:
            raise SystemExit(
                "Fail-closed: vLLM pass@8 requires T06-merged base when lineage "
                f"has instruction_adapter={instruction!r}, but merge failed: {exc}. "
                "Use --backend hf (lineage-correct) or fix the merge."
            ) from exc

    print(
        f"[pass8-vllm] base={vllm_base} (raw_base={base}) instruction={instruction} "
        f"adapter={sft_checkpoint} gpu_mem={gpu_memory_utilization} "
        f"max_num_seqs={max_num_seqs} max_model_len={max_model_len}",
        flush=True,
    )
    llm = LLM(
        model=vllm_base,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=int(max_lora_rank),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        enable_prefix_caching=True,
        disable_log_stats=False,
        # Avoid long silent CUDA-graph capture hangs on fresh 5090 stacks.
        enforce_eager=True,
    )
    lora_request = LoRARequest("sft_pass8", 1, str(sft_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(vllm_base, trust_remote_code=True)
    return llm, lora_request, tokenizer, ckpt_hash, vllm_base


def _calibrate_vllm_batched(
    shard_rows: list[dict[str, Any]],
    *,
    llm: Any,
    lora_request: Any,
    tokenizer: Any,
    system_prompt: str,
    verify_fn: Callable[[dict[str, Any], str], bool],
    n: int,
    base_seed: int,
    checkpoint_hash: str,
    tokenizer_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_batch_size: int,
) -> list[PassAtNResult]:
    """Continuous-batched pass@n via vLLM (n completions per prompt)."""
    from vllm import SamplingParams

    sys_hash = _hash_text(system_prompt) if system_prompt else ""
    rendered = [
        _apply_chat(tokenizer, system_prompt, str(prob.get("prompt", "")))
        for prob in shard_rows
    ]
    results: list[PassAtNResult] = []
    t0 = time.time()
    total = len(shard_rows)
    done = 0
    batch = max(1, int(prompt_batch_size))

    for start in range(0, total, batch):
        chunk_probs = shard_rows[start : start + batch]
        chunk_prompts = rendered[start : start + batch]
        # Match HF contract: parent seed = seeds[0] = base_seed + i*n, n return seqs.
        sampling = [
            SamplingParams(
                n=int(n),
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_new_tokens),
                seed=int(base_seed + (start + j) * n),
            )
            for j in range(len(chunk_prompts))
        ]
        outputs = llm.generate(
            chunk_prompts, sampling, lora_request=lora_request, use_tqdm=True
        )
        for j, (prob, out) in enumerate(zip(chunk_probs, outputs)):
            i = start + j
            seeds = [base_seed + i * n + k for k in range(n)]
            completions = [o.text for o in out.outputs]
            if len(completions) != n:
                raise SystemExit(
                    f"vLLM returned {len(completions)} completions, expected {n} "
                    f"for {prob.get('problem_id')}"
                )
            flags = [bool(verify_fn(prob, c)) for c in completions]
            passes = sum(flags)
            frac = passes / n if n else 0.0
            results.append(
                PassAtNResult(
                    problem_id=str(prob.get("problem_id") or prob.get("id") or i),
                    n=n,
                    passes=passes,
                    pass_fraction=frac,
                    band=assign_band(frac),
                    seeds=seeds,
                    checkpoint_hash=checkpoint_hash,
                    tokenizer_id=tokenizer_id,
                    system_prompt_hash=sys_hash,
                    raw=flags,
                )
            )
        done += len(chunk_probs)
        elapsed = max(time.time() - t0, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"[pass8-vllm] {done}/{total} prompts  {rate:.2f} prompts/s  "
            f"ETA {eta/60:.1f} min  (n={n} => {rate*n:.1f} samples/s)",
            flush=True,
        )
    return results


def _stamp_candidates(
    shard_rows: list[dict[str, Any]], results: list[PassAtNResult]
) -> list[dict[str, Any]]:
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
        # Keep difficulty_tag inside metadata — top-level breaks accept_rlvr_row
        # schema (additionalProperties=false).
        meta = dict(out.get("metadata") or {})
        meta["difficulty_tag"] = band
        meta["pass_at_8"] = {
            "n": cal.n,
            "passes": cal.passes,
            "pass_fraction": cal.pass_fraction,
        }
        meta["difficulty_band"] = band
        meta["difficulty_source"] = "sft_pass_at_8"
        out["metadata"] = meta
        stamped.append(out)
    return stamped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-calibration", type=Path, required=True)
    parser.add_argument("--out-candidates", type=Path, required=True,
                        help="Candidates stamped with empirical_difficulty")
    parser.add_argument("--sft-checkpoint", type=Path, default=None)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stub", action="store_true",
                        help="Credential/GPU-free stub generator for tests")
    parser.add_argument(
        "--backend",
        choices=("hf", "vllm"),
        default="hf",
        help="Generation backend (vllm = continuous batching, recommended on 5090)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-lora-rank", type=int, default=128)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=0,
        help="Prompts per generate step (each expands to n samples). "
        "0 = backend default (hf:16, vllm:256)",
    )
    parser.add_argument(
        "--merge-sft-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge SFT LoRA into weights before HF generate (faster; same lineage)",
    )
    args = parser.parse_args()
    if args.prompt_batch_size <= 0:
        # HF: 4 prompts × n=8 = 32 seqs — saturates 5090 without KV OOM thrash.
        args.prompt_batch_size = 256 if args.backend == "vllm" else 4

    rows = _read_jsonl(args.candidates)
    if args.num_shards < 1:
        raise SystemExit("num_shards must be >= 1")
    shard_rows = [
        r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id
    ]
    print(
        f"[pass8] shard={args.shard_id}/{args.num_shards} "
        f"rows={len(shard_rows)} n={args.n} backend={args.backend}",
        flush=True,
    )

    system_prompt = args.system_prompt
    if args.stub:
        from rlvr_synth.calibration.pass_at_n import stub_generate_fn_factory

        generate_fn = stub_generate_fn_factory()
        results = calibrate_pass_at_n(
            shard_rows,
            generate_fn=generate_fn,
            verify_fn=_make_verify_fn(),
            n=args.n,
            base_seed=args.base_seed,
            checkpoint_hash="stub",
            tokenizer_id="stub",
            system_prompt=system_prompt,
        )
        checkpoint_hash = "stub"
    else:
        if args.sft_checkpoint is None:
            raise SystemExit("--sft-checkpoint is required unless --stub")
        if args.backend == "vllm":
            llm, lora_request, tokenizer, checkpoint_hash, tokenizer_id = (
                _make_vllm_engine(
                    sft_checkpoint=args.sft_checkpoint,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=args.max_num_seqs,
                    max_lora_rank=args.max_lora_rank,
                )
            )
            results = _calibrate_vllm_batched(
                shard_rows,
                llm=llm,
                lora_request=lora_request,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                verify_fn=_make_verify_fn(),
                n=args.n,
                base_seed=args.base_seed,
                checkpoint_hash=checkpoint_hash,
                tokenizer_id=tokenizer_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompt_batch_size=args.prompt_batch_size,
            )
        else:
            model, tokenizer, checkpoint_hash, _base, _instr = (
                _load_hf_lineage_for_infer(
                    sft_checkpoint=args.sft_checkpoint,
                    merge_sft_lora=bool(args.merge_sft_lora),
                )
            )
            tokenizer_id = getattr(tokenizer, "name_or_path", "") or _base
            results = _calibrate_hf_batched(
                shard_rows,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                verify_fn=_make_verify_fn(),
                n=args.n,
                base_seed=args.base_seed,
                checkpoint_hash=checkpoint_hash,
                tokenizer_id=tokenizer_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompt_batch_size=max(1, int(args.prompt_batch_size)),
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
    stamped = _stamp_candidates(shard_rows, results)
    _write_jsonl(args.out_candidates, stamped)
    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_candidates": len(shard_rows),
        "backend": args.backend,
        "bands": {
            b: sum(1 for r in results if r.band == b)
            for b in ("mastered", "easy", "medium", "hard", "deferred")
        },
        "checkpoint_hash": checkpoint_hash,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
