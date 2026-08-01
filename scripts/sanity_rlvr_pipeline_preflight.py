#!/usr/bin/env python
"""Pre-full-run sanity checks for rlvr_pipeline (no GPU train). Writes debug-a273d4.log."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
LOG = Path(r"c:\Users\Azooo\arabic-reasoning-rlvr-sota\debug-a273d4.log")
sys.path.insert(0, str(REPO / "src"))


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "a273d4",
                    "hypothesisId": hypothesis_id,
                    "runId": "sanity",
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> int:
    failures: list[str] = []
    from rlvr_pipeline.config import RLVRConfig
    from rlvr_pipeline.rewards import ALL_REWARD_FUNCS, REWARD_FUNCS_ORDER, diagnose_reward_weights
    from rlvr_pipeline.data import _derive_difficulty
    from rlvr_pipeline.curriculum_sampler import CurriculumSampler, BUCKET_NAMES
    from rlvr_pipeline.hub_checkpoint_callback import HubCheckpointCallback, resolve_hf_token
    from rlvr_pipeline.failure_mining_trainer import GRPOTrainerWithFailureMining
    import torch

    yaml_path = REPO / "configs" / "qwen_4b_2x5090_v4_sota.yaml"
    cfg = RLVRConfig.from_yaml(yaml_path)
    _log("A", "sanity:config_load", "v4 yaml loaded", {
        "zero_variance": cfg.zero_variance_strategy,
        "curriculum": cfg.curriculum_schedule_type,
        "top_entropy_quantile": cfg.top_entropy_quantile,
        "push_hub": cfg.push_checkpoints_to_hub,
        "hub_id": cfg.hub_model_id,
        "keep_via_save_total_limit": cfg.save_total_limit,
        "delete_after_push": cfg.delete_local_checkpoint_after_hub_push,
        "sft_lr": cfg.sft_learning_rate,
        "grpo_lr": cfg.learning_rate,
        "reward_weights": cfg.reward_weights,
        "coldstart_exists": Path(cfg.coldstart_data_path).exists(),
        "train_exists": Path(cfg.train_data_path).exists() if cfg.train_data_path else False,
        "eval_exists": Path(cfg.eval_data_path).exists() if cfg.eval_data_path else False,
    })

    grpo = cfg.build_grpo_config(include_model_init=False)
    teq = getattr(grpo, "top_entropy_quantile", None)
    if teq != cfg.top_entropy_quantile:
        failures.append(f"top_entropy_quantile mismatch cfg={cfg.top_entropy_quantile} grpo={teq}")
    _log("A", "sanity:grpo_built", "GRPOConfig built", {
        "top_entropy_quantile": teq,
        "beta": grpo.beta,
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "report_to": getattr(grpo, "report_to", None),
    })

    # E: reward order
    names_from_funcs = [f.__name__.replace("_reward_func", "").replace("_penalty_func", "") for f in ALL_REWARD_FUNCS]
    order_ok = list(REWARD_FUNCS_ORDER) == [
        "correctness", "format", "language", "answer_leak", "structural_leak", "length"
    ]
    if len(ALL_REWARD_FUNCS) != 6 or len(cfg.reward_weights) != 6:
        failures.append("reward funcs/weights length != 6")
    diagnose_reward_weights(cfg.reward_weights)
    _log("E", "sanity:rewards", "reward contract", {
        "REWARD_FUNCS_ORDER": list(REWARD_FUNCS_ORDER),
        "func_names": [f.__name__ for f in ALL_REWARD_FUNCS],
        "order_ok": order_ok,
        "weights": cfg.reward_weights,
    })

    # B: difficulty derivation + curriculum orphan tags
    samples = [
        {"difficulty_tag": "easy"},
        {"difficulty_tag": "mastered", "empirical_difficulty": {"band": "mastered"}},
        {"empirical_difficulty": {"band": "deferred"}},
        {"num_steps": 1},
        {"difficulty_tag": "hard"},
    ]
    derived = [_derive_difficulty(s) for s in samples]
    orphan = [t for t in derived if t not in set(BUCKET_NAMES)]
    tags = ["easy", "easy", "medium", "hard", "mastered", "deferred", "trivial"]
    # After fix, _derive_difficulty must never emit non-bucket tags.
    derived_from_tags = [
        _derive_difficulty({"difficulty_tag": t, "empirical_difficulty": {"band": t}})
        for t in ("mastered", "deferred", "easy")
    ]
    sampler = CurriculumSampler(tags, total_steps=100, schedule_type="gaussian", sigma_fraction=0.2, seed=0)
    bucket_sizes = {b: len(sampler._bucket_indices[b]) for b in BUCKET_NAMES}
    unreachable = sum(1 for t in tags if t not in set(BUCKET_NAMES))
    _log("B", "sanity:curriculum", "difficulty/curriculum buckets", {
        "derived_samples": derived,
        "orphan_derived": orphan,
        "derived_from_mastered_deferred": derived_from_tags,
        "bucket_sizes": bucket_sizes,
        "unreachable_tag_count": unreachable,
        "BUCKET_NAMES": list(BUCKET_NAMES),
    })
    if orphan:
        failures.append(f"_derive_difficulty can emit non-curriculum tags: {orphan}")
    if derived_from_tags[:2] != ["easy", "hard"]:
        failures.append(f"mastered/deferred mapping expected [easy,hard], got {derived_from_tags[:2]}")

    # C: ZVP slice on fake trainer
    trainer = object.__new__(GRPOTrainerWithFailureMining)
    trainer.zero_variance_strategy = "direct_scoring"
    num_gen, num_groups = 4, 2
    advantages = torch.zeros(num_gen)  # local rank owns group 0 only
    rewards = torch.tensor([0.0] * num_gen + [1.0] * num_gen)  # group0 all wrong flat, group1 all correct flat
    # rewards_per_func: [N, 6], correctness col 0
    rpf = torch.zeros(num_gen * num_groups, 6)
    rpf[:num_gen, 0] = 0.0
    rpf[num_gen:, 0] = 1.0
    out = {"advantages": advantages.clone()}
    # process owns indices 0..num_gen-1 (group 0)
    trainer._process_failure_mining(
        out, rpf, rewards, advantages, num_gen, num_groups,
        process_start=0, process_end=num_gen,
    )
    zvp_vals = out["advantages"].tolist()
    _log("C", "sanity:zvp", "ZVP process_mining result", {
        "advantages": zvp_vals,
        "stats": getattr(trainer, "_failure_mining_stats", {}),
        "expect_all_neg1_local": all(v == -1.0 for v in zvp_vals),
    })
    if not all(v == -1.0 for v in zvp_vals):
        failures.append(f"ZVP expected local advantages=-1, got {zvp_vals}")

    # D: hub delete keep_n vs disk relief
    token_present = bool(resolve_hf_token())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for step in (50, 100, 150, 200):
            (root / f"checkpoint-{step}").mkdir()
            (root / f"checkpoint-{step}" / "adapter_config.json").write_text("{}", encoding="utf-8")
        cb = HubCheckpointCallback(
            hub_model_id="test/dummy",
            delete_local_after_push=True,
            keep_local_last_n=getattr(cfg, "hub_keep_local_last_n", 1),
        )
        cb._pushed_steps = {50, 100, 150, 200}
        before = sorted(p.name for p in root.iterdir() if p.is_dir())
        cb._delete_older_local_checkpoints(str(root))
        after = sorted(p.name for p in root.iterdir() if p.is_dir())
        _log("D", "sanity:hub_delete", "delete-after-push with hub_keep_local_last_n", {
            "keep_n": getattr(cfg, "hub_keep_local_last_n", 1),
            "before": before,
            "after": after,
            "deleted": sorted(set(before) - set(after)),
            "token_env_present": token_present,
        })
        if after != ["checkpoint-200"]:
            failures.append(
                f"hub_keep_local_last_n should leave only checkpoint-200, got {after}"
            )

    # Extended trainer selection flags
    use_ext = (
        cfg.zero_variance_strategy != "discard"
        or cfg.enable_crps
        or cfg.curriculum_schedule_type != "none"
    )
    _log("A", "sanity:trainer_flags", "extended trainer gate", {
        "use_extended_trainer": use_ext,
        "zv": cfg.zero_variance_strategy,
        "curriculum": cfg.curriculum_schedule_type,
        "crps": cfg.enable_crps,
    })
    if not use_ext:
        failures.append("expected extended trainer for ZVP+curriculum")

    # Coldstart load smoke (first 3 rows path exists)
    cold = Path(cfg.coldstart_data_path)
    if cold.exists():
        from rlvr_pipeline.data import load_cold_start_sft_dataset
        try:
            # Prefer a tiny sample if file huge — still load via API
            ds = load_cold_start_sft_dataset(str(cold), system_prompt=cfg.system_prompt)
            _log("A", "sanity:coldstart", "coldstart loaded", {"n": len(ds), "cols": list(ds.column_names)[:12]})
        except Exception as e:
            failures.append(f"coldstart load failed: {e}")
            _log("A", "sanity:coldstart", "coldstart load FAILED", {"error": str(e)})
    else:
        _log("A", "sanity:coldstart", "coldstart missing", {"path": str(cold)})

    # Candidates difficulty if present
    cand = REPO / "data" / "arabic_reasoning_rlvr_candidates_v5.jsonl"
    if cand.exists():
        tags_c = []
        with open(cand, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 200:
                    break
                row = json.loads(line)
                meta = row.get("metadata") or {}
                if "difficulty_tag" in row:
                    meta = {**meta, "difficulty_tag": row["difficulty_tag"]}
                tags_c.append(_derive_difficulty(meta))
        _log("B", "sanity:candidates_tags", "first 200 candidate derived tags", {
            "counts": dict(Counter(tags_c)),
            "orphan": dict(Counter(t for t in tags_c if t not in set(BUCKET_NAMES))),
        })

    summary = {"ok": not failures, "failures": failures}
    _log("A", "sanity:summary", "preflight summary", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
