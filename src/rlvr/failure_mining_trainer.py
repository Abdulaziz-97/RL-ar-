"""
GRPOTrainer subclass with failure mining integration.

Injects RL-ZVP (direct confidence-scaled scoring) for zero-variance groups,
POPO (prioritized replay buffer) as secondary mode, and CRPS (progressive
suffix hints) for the hard curriculum stage.

All three methodologies are switchable via config flags -- no code change needed.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
from torch.utils.data import Sampler

from rlvr.failure_mining import (
    ReplayBuffer,
    ReplayEntry,
    is_zero_variance_group,
    is_effective_group,
    maybe_push_to_buffer,
)
from rlvr.crps import (
    SuccessTrace,
    SuccessTraceStore,
    find_related_success_trace,
    build_progressive_suffix,
    get_crps_suffix_fraction,
)
from trl import GRPOTrainer


class GRPOTrainerWithFailureMining(GRPOTrainer):
    """GRPOTrainer with RL-ZVP / POPO / CRPS failure mining integration.

    Config flags (set via SOTAConfig):
        zero_variance_strategy: "direct_scoring" | "replay_buffer" | "discard"
        enable_crps: bool
    """

    def __init__(self, *args, zero_variance_strategy: str = "direct_scoring",
                 replay_buffer_size: int = 512, enable_crps: bool = True,
                 crps_max_age_steps: int = 100, crps_max_traces: int = 256,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_variance_strategy = zero_variance_strategy
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.enable_crps = enable_crps
        self.crps_max_age_steps = crps_max_age_steps
        self.success_trace_store = SuccessTraceStore(max_traces=crps_max_traces)
        self._failure_mining_stats: dict[str, float] = {}
        self._curriculum_sampler: Optional[Sampler] = None
        self._curriculum_config: Optional[dict] = None
        self._failure_counter: dict[str, int] = {}

    def attach_curriculum_sampler(self, config, dataset) -> None:
        from rlvr_sota.curriculum_sampler import CurriculumSampler, CurriculumCallback

        difficulty_tags = list(dataset["difficulty_tag"])
        batch_size = config.per_device_train_batch_size
        steps_per_epoch = max(len(dataset) // max(batch_size * config.gradient_accumulation_steps, 1), 1)
        total_steps = steps_per_epoch * config.num_train_epochs

        sampler = CurriculumSampler(
            difficulty_tags=difficulty_tags,
            total_steps=total_steps,
            schedule_type=config.curriculum_schedule_type,
            sigma_fraction=config.sigma_fraction,
            seed=config.seed,
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

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.steps_per_generation,
            sampler_fn=sampler_fn,
            is_training=True,
        )

    def _generate_and_score_completions(self, inputs):
        inputs = self._inject_crps_hints(inputs)

        output = super()._generate_and_score_completions(inputs)

        completion_ids = output["completion_ids"]
        prompt_ids = output["prompt_ids"]
        advantages = output["advantages"]
        completion_mask = output.get("completion_mask")
        old_per_token_logps = output.get("old_per_token_logps")

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completion_ids_list = [ids.tolist() for ids in completion_ids]

        rewards_per_func = self._calculate_rewards(inputs, prompts_text, completions_text, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)

        num_gen = self.num_generations
        B_val = rewards.shape[0]
        num_groups = B_val // num_gen

        self._process_failure_mining(
            output, rewards_per_func, rewards, advantages,
            completion_mask, old_per_token_logps,
            num_gen, num_groups,
        )

        if self.enable_crps and self.state.global_step > 0:
            self._maybe_record_success_traces(completion_ids, rewards, inputs, num_gen, num_groups)

        return output

    def _inject_crps_hints(self, inputs) -> list:
        if not self.enable_crps or len(self.success_trace_store) == 0:
            return inputs

        traces = self.success_trace_store.traces
        current_step = self.state.global_step
        result = list(inputs)

        for i, inp in enumerate(result):
            difficulty = inp.get("difficulty_tag", "medium")
            if difficulty != "hard":
                continue

            domain = inp.get("domain", "")
            puzzle_type = inp.get("puzzle_type", domain) or domain
            failure_key = f"{puzzle_type}:{difficulty}"

            trace = find_related_success_trace(
                traces, puzzle_type=puzzle_type, difficulty_tag=difficulty,
                max_age_steps=self.crps_max_age_steps, current_step=current_step,
            )
            if trace is None:
                continue

            failure_count = self._failure_counter.get(failure_key, 0)
            fraction = get_crps_suffix_fraction(failure_count + 1)
            if fraction <= 0.0:
                continue

            suffix = build_progressive_suffix(trace, fraction)
            if not suffix:
                continue

            prompt_messages = inp.get("prompt", [])
            if not (prompt_messages and isinstance(prompt_messages[-1], dict)):
                continue

            inp = result[i] = copy.deepcopy(inp)
            inp["prompt"][-1]["content"] = (
                inp["prompt"][-1].get("content", "")
                + "\n\nHint (partial solution):\n" + suffix
            )
            self._failure_counter[failure_key] = failure_count + 1

        return result

    def _process_failure_mining(self, output, rewards_per_func, rewards,
                                 advantages, completion_mask, old_per_token_logps,
                                 num_gen, num_groups):
        zero_var_count = 0

        if self.zero_variance_strategy == "direct_scoring":
            if old_per_token_logps is not None and completion_mask is not None:
                for g in range(num_groups):
                    start = g * num_gen
                    end = start + num_gen

                    group_correctness = rewards_per_func[start:end, 0].tolist()
                    group_format = rewards_per_func[start:end, 1].tolist()
                    group_rewards = rewards[start:end].tolist()

                    is_correctness_zero = is_zero_variance_group(group_correctness) and abs(group_correctness[0]) < 1e-6
                    is_format_zero = is_zero_variance_group(group_format) and abs(group_format[0]) < 1e-6
                    is_composite_zero = is_zero_variance_group(group_rewards)

                    if is_correctness_zero or is_format_zero or is_composite_zero:
                        zero_var_count += 1
                        all_wrong = (rewards_per_func[start:end, 0] < 0.5).all().item()
                        group_logps = old_per_token_logps[start:end]
                        confidences = torch.exp(group_logps).float()

                        if is_correctness_zero:
                            signals = -confidences
                        elif all_wrong:
                            signals = -confidences
                        else:
                            signals = 1.0 - confidences

                        signals = signals * completion_mask[start:end].float()
                        mask_sums = completion_mask[start:end].sum(dim=1).clamp(min=1)
                        scalar_advantages = signals.sum(dim=1) / mask_sums

                        for j in range(end - start):
                            advantages[start + j] = scalar_advantages[j]

                output["advantages"] = advantages

        elif self.zero_variance_strategy == "replay_buffer":
            for g in range(num_groups):
                start = g * num_gen
                end = start + num_gen
                group_rewards = rewards[start:end].tolist()

                if is_effective_group(group_rewards):
                    if old_per_token_logps is not None:
                        entry = ReplayEntry(
                            prompt_id="",
                            completions=[],
                            rewards=group_rewards,
                            old_policy_log_probs=old_per_token_logps[start:end].cpu().tolist(),
                            step_generated=self.state.global_step,
                        )
                        maybe_push_to_buffer(self.replay_buffer, entry)
                else:
                    zero_var_count += 1

        effective_count = num_groups - zero_var_count
        total_groups = max(num_groups, 1)

        self._failure_mining_stats = {
            "effective_sample_ratio": effective_count / total_groups,
            "zero_variance_group_count": float(zero_var_count),
            "replay_buffer_size": float(len(self.replay_buffer)),
        }

    def _maybe_record_success_traces(self, completion_ids, rewards, inputs,
                                      num_gen, num_groups):
        if completion_ids is None:
            return

        for g in range(num_groups):
            start = g * num_gen
            end = start + num_gen
            group_rewards = rewards[start:end].tolist()
            group_inputs = inputs[start:end]

            for i, r in enumerate(group_rewards):
                if abs(r - 1.0) < 1e-6:
                    idx = start + i
                    token_ids = completion_ids[idx].tolist()
                    decoded_tokens = [
                        self.processing_class.decode([t])
                        for t in token_ids
                    ]
                    difficulty = group_inputs[i].get("difficulty_tag", "hard")
                    puzzle = group_inputs[i].get("puzzle_type", "") or group_inputs[i].get("domain", "")
                    trace = SuccessTrace(
                        prompt_id=group_inputs[i].get("sample_id", ""),
                        puzzle_type=puzzle,
                        difficulty_tag=difficulty,
                        token_sequence=decoded_tokens,
                        step_recorded=self.state.global_step,
                    )
                    self.success_trace_store.add(trace)
                    break

    def log_metrics(self, split, metrics, **kwargs):
        merged = {**metrics, **self._failure_mining_stats}
        if self.enable_crps:
            merged["crps_trace_count"] = float(len(self.success_trace_store))
        return super().log_metrics(split, merged, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if self.zero_variance_strategy != "discard" and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            for key, val in self._failure_mining_stats.items():
                self.log({f"failure_mining/{key}": val})

        return loss
