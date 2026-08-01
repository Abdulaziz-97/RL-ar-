"""
GRPOTrainer subclass with failure mining integration.

Injects direct confidence-scaled scoring for zero-variance groups and CRPS
(progressive suffix hints) for the hard curriculum stage.

Both methodologies are switchable via config flags.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Sampler

from rlvr.failure_mining import is_zero_variance_group
from rlvr.crps import (
    SuccessTrace,
    SuccessTraceStore,
    find_related_success_trace,
    build_progressive_suffix,
    get_crps_suffix_fraction,
)
from trl import GRPOTrainer

CORRECTNESS_IDX = 0
FORMAT_IDX = 1


class GRPOTrainerWithFailureMining(GRPOTrainer):
    """GRPOTrainer with direct zero-variance scoring and CRPS integration.

    Config flags (set via RLVRConfig):
        zero_variance_strategy: "direct_scoring" | "discard"
        enable_crps: bool
    """

    def __init__(self, *args, zero_variance_strategy: str = "direct_scoring",
                 enable_crps: bool = True,
                 crps_max_age_steps: int = 100, crps_max_traces: int = 256,
                 **kwargs):
        if zero_variance_strategy not in {"direct_scoring", "discard"}:
            raise ValueError(
                "zero_variance_strategy must be 'direct_scoring' or 'discard'; "
                "'replay_buffer' is not implemented safely in the trainer"
            )
        super().__init__(*args, **kwargs)
        if enable_crps and self.accelerator.num_processes > 1:
            raise ValueError(
                "CRPS is rank-local and cannot preserve identical prompts for a "
                "generation group under DDP; disable CRPS for multi-GPU training"
            )
        self.zero_variance_strategy = zero_variance_strategy
        self.enable_crps = enable_crps
        self.crps_max_age_steps = crps_max_age_steps
        self.success_trace_store = SuccessTraceStore(max_traces=crps_max_traces)
        self._failure_mining_stats: dict[str, float] = {}
        self._curriculum_sampler: Optional[Sampler] = None
        self._curriculum_config: Optional[dict] = None
        self._failure_counter: dict[str, int] = {}

    def attach_curriculum_sampler(self, config, dataset) -> None:
        from rlvr_pipeline.curriculum_sampler import CurriculumSampler, CurriculumCallback

        difficulty_tags = list(dataset["difficulty_tag"])
        batch_size = config.per_device_train_batch_size
        steps_per_epoch = max(len(dataset) // max(batch_size * config.gradient_accumulation_steps, 1), 1)
        total_steps = (
            config.max_steps
            if config.max_steps is not None and config.max_steps > 0
            else max(int(steps_per_epoch * config.num_train_epochs), 1)
        )

        gen_batch = getattr(self.args, "generation_batch_size", getattr(self.args, "per_device_train_batch_size", 4) * getattr(self.args, "gradient_accumulation_steps", 1))
        batch_size = max(1, gen_batch // max(1, self.num_generations))
        num_iters = getattr(self, "num_iterations", 1)
        sampler = CurriculumSampler(
            difficulty_tags=difficulty_tags,
            total_steps=total_steps,
            schedule_type=config.curriculum_schedule_type,
            sigma_fraction=config.sigma_fraction,
            seed=config.seed,
            mini_repeat_count=self.num_generations,
            batch_size=batch_size,
            repeat_count=num_iters * getattr(self.args, "steps_per_generation", 1),
        )
        self._curriculum_sampler = sampler
        self._curriculum_config = {
            "schedule_type": config.curriculum_schedule_type,
            "sigma_fraction": config.sigma_fraction,
        }

        self.add_callback(CurriculumCallback(sampler))

    def get_train_dataloader(self):
        if self._curriculum_sampler is not None:
            sampler_fn = lambda _dataset: self._curriculum_sampler  # noqa: E731
        else:
            sampler_fn = self._get_train_sampler

        steps_per_gen = getattr(self.args, "steps_per_generation", 1)
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * steps_per_gen,
            sampler_fn=sampler_fn,
            is_training=True,
        )

    def _generate_and_score_completions(self, inputs):
        inputs = self._inject_crps_hints(inputs)

        output = super()._generate_and_score_completions(inputs)

        completion_ids = output["completion_ids"]
        prompt_ids = output["prompt_ids"]
        advantages = output["advantages"]

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completion_ids_list = [ids.tolist() for ids in completion_ids]

        global_rewards_per_func = self._calculate_rewards(
            inputs, prompts_text, completions_text, completion_ids_list
        )
        local_count = len(inputs)
        process_index = self.accelerator.process_index
        process_start = process_index * local_count
        process_end = process_start + local_count
        local_rewards_per_func = global_rewards_per_func[process_start:process_end]
        global_rewards = (
            global_rewards_per_func
            * self.reward_weights.to(global_rewards_per_func.device).unsqueeze(0)
        ).nansum(dim=1)

        num_gen = self.num_generations
        num_groups = global_rewards.shape[0] // num_gen

        # Explicitly log all rollouts & rewards to completions_log.jsonl (rank-safe, fail visible)
        try:
            if self.accelerator.is_main_process:
                log_file = Path(self.args.output_dir) / "completions_log.jsonl"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    for p, c, r in zip(prompts_text, completions_text, global_rewards.tolist()):
                        f.write(
                            json.dumps(
                                {
                                    "step": self.state.global_step,
                                    "prompt": p,
                                    "completion": c,
                                    "reward": r,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        except Exception as e:
            raise RuntimeError(
                f"Failed to write completions log under {self.args.output_dir}: {e}"
            ) from e

        self._process_failure_mining(
            output,
            global_rewards_per_func,
            global_rewards,
            advantages,
            num_gen,
            num_groups,
            process_start,
            process_end,
        )

        if self.enable_crps and self.state.global_step > 0:
            self._maybe_record_success_traces(
                completion_ids, local_rewards_per_func, inputs,
            )

        return output

    @staticmethod
    def _crps_failure_key(inp: dict) -> str:
        sample_id = inp.get("sample_id", "") or ""
        domain = inp.get("domain", "")
        puzzle_type = inp.get("puzzle_type", domain) or domain
        difficulty = inp.get("difficulty_tag", "medium")
        return f"{sample_id}:{puzzle_type}:{difficulty}"

    def _inject_crps_hints(self, inputs) -> list:
        if not self.enable_crps or len(self.success_trace_store) == 0:
            return inputs

        traces = self.success_trace_store.traces
        current_step = self.state.global_step
        result = list(inputs)

        grouped_indices: dict[str, list[int]] = {}
        for i, inp in enumerate(inputs):
            grouped_indices.setdefault(self._crps_failure_key(inp), []).append(i)

        for failure_key, indices in grouped_indices.items():
            first_index = indices[0]
            inp = inputs[first_index]
            difficulty = inp.get("difficulty_tag", "medium")
            if difficulty != "hard":
                continue

            domain = inp.get("domain", "")
            puzzle_type = inp.get("puzzle_type", domain) or domain

            trace = find_related_success_trace(
                traces, puzzle_type=puzzle_type, difficulty_tag=difficulty,
                max_age_steps=self.crps_max_age_steps, current_step=current_step,
                prompt_id=str(inp.get("sample_id", "") or ""),
            )
            if trace is None:
                continue

            failure_count = self._failure_counter.get(failure_key, 0)
            fraction = get_crps_suffix_fraction(failure_count)
            if fraction <= 0.0:
                self._failure_counter[failure_key] = failure_count + 1
                continue

            suffix = build_progressive_suffix(trace, fraction)
            if not suffix:
                continue

            prompt_messages = inp.get("prompt", [])
            if not (prompt_messages and isinstance(prompt_messages[-1], dict)):
                continue

            modified = copy.deepcopy(inp)
            modified["prompt"][-1]["content"] = (
                modified["prompt"][-1].get("content", "")
                + "\n\nتلميح من محاولة سابقة للمسألة نفسها:\n" + suffix
            )
            for i in indices:
                result[i] = copy.deepcopy(modified)
            self._failure_counter[failure_key] = failure_count + 1

        return result

    def _process_failure_mining(
        self,
        output,
        rewards_per_func,
        rewards,
        advantages,
        num_gen,
        num_groups,
        process_start=0,
        process_end=None,
    ):
        process_end = process_end if process_end is not None else process_start + len(advantages)
        zero_var_count = 0

        for g in range(num_groups):
            start = g * num_gen
            end = start + num_gen
            group_rewards = rewards[start:end].tolist()

            if is_zero_variance_group(group_rewards):
                zero_var_count += 1
                if self.zero_variance_strategy == "direct_scoring":
                    # Preserve normal GRPO advantages whenever auxiliary rewards
                    # vary. For truly flat groups, apply a bounded signal.
                    correctness = rewards_per_func[start:end, CORRECTNESS_IDX]
                    all_wrong = (correctness < 0.5).all().item()
                    all_correct = (correctness >= 0.5).all().item()
                    if not (all_wrong or all_correct):
                        continue
                    overlap_start = max(start, process_start)
                    overlap_end = min(end, process_end)
                    if overlap_start < overlap_end:
                        local_start = overlap_start - process_start
                        local_end = overlap_end - process_start
                        advantages[local_start:local_end] = -1.0 if all_wrong else 1.0
                        # #region agent log
                        try:
                            import json as _json, time as _time
                            from pathlib import Path as _Path
                            _log = _Path(r"c:\Users\Azooo\arabic-reasoning-rlvr-sota\debug-a273d4.log")
                            with open(_log, "a", encoding="utf-8") as _f:
                                _f.write(_json.dumps({
                                    "sessionId": "a273d4", "hypothesisId": "C", "runId": "sanity",
                                    "location": "failure_mining_trainer.py:zvp_apply",
                                    "message": "ZVP direct_scoring applied",
                                    "data": {
                                        "group": g, "all_wrong": all_wrong, "all_correct": all_correct,
                                        "process_start": process_start, "process_end": process_end,
                                        "local_start": local_start, "local_end": local_end,
                                        "adv_value": -1.0 if all_wrong else 1.0,
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                }, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        # #endregion

        # Refinement: Apply advantage clamping [-5.0, 5.0] to prevent low-sigma advantage amplification spikes
        raw_advantages = advantages.clone()
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)
        clamped_count = (advantages != raw_advantages).sum().item()
        total_adv_elements = max(advantages.numel(), 1)

        output["advantages"] = advantages

        effective_count = num_groups - zero_var_count
        total_groups = max(num_groups, 1)

        self._failure_mining_stats = {
            "effective_sample_ratio": effective_count / total_groups,
            "zero_variance_group_count": float(zero_var_count),
            "frac_advantages_clamped": float(clamped_count / total_adv_elements),
        }

    def _maybe_record_success_traces(self, completion_ids, rewards_per_func, inputs):
        if completion_ids is None:
            return

        recorded_ids: set[str] = set()
        for idx, inp in enumerate(inputs):
            sample_id = str(inp.get("sample_id", "") or "")
            if not sample_id or sample_id in recorded_ids:
                continue
            if (
                rewards_per_func[idx, CORRECTNESS_IDX] >= 1.0 - 1e-6
                and rewards_per_func[idx, FORMAT_IDX] > 0.0
            ):
                token_ids = completion_ids[idx].tolist()
                full_text = self.processing_class.decode(token_ids, skip_special_tokens=True)
                # Never put a prior final answer into a future prompt.
                think_start = full_text.find("<think>")
                think_end = full_text.find("</think>")
                if think_start >= 0 and think_end > think_start:
                    full_text = full_text[think_start + len("<think>") : think_end]
                elif "<answer>" in full_text:
                    full_text = full_text.split("<answer>", 1)[0]
                words = full_text.split()
                if not words:
                    continue
                difficulty = inp.get("difficulty_tag", "hard")
                puzzle = inp.get("puzzle_type", "") or inp.get("domain", "")
                trace = SuccessTrace(
                    prompt_id=sample_id,
                    puzzle_type=puzzle,
                    difficulty_tag=difficulty,
                    token_sequence=words,
                    step_recorded=self.state.global_step,
                )
                self.success_trace_store.add(trace)
                recorded_ids.add(sample_id)
                failure_key = self._crps_failure_key(inp)
                self._failure_counter.pop(failure_key, None)

    def log_metrics(self, split, metrics, **kwargs):
        merged = {**metrics, **self._failure_mining_stats}
        if self.enable_crps:
            merged["crps_trace_count"] = float(len(self.success_trace_store))
        return super().log_metrics(split, merged, **kwargs)

    def compute_loss(self, model, inputs, *args, **kwargs):
        loss = super().compute_loss(model, inputs, *args, **kwargs)

        if self.zero_variance_strategy != "discard" and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            for key, val in self._failure_mining_stats.items():
                self.log({f"failure_mining/{key}": val})

        return loss
