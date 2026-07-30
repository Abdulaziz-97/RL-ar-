"""Live training stability guards for GRPO (format/entropy, adaptive beta/temp)."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rlvr.monitoring import MonitoringDashboard

logger = logging.getLogger(__name__)

EntropyAction = Literal["warn", "reduce_lr", "stop"]


class StabilityCallback(TrainerCallback):
    """Monitor format/entropy collapse, adaptive KL beta, and early length diagnostics."""

    def __init__(
        self,
        *,
        consecutive: int = 2,
        entropy_action: EntropyAction = "reduce_lr",
        lr_reduction_factor: float = 0.5,
        dashboard: Optional[MonitoringDashboard] = None,
        # Adaptive beta when KL≈0 but entropy is healthy
        enable_adaptive_beta: bool = True,
        kl_near_zero_threshold: float = 1e-3,
        stuck_beta: float = 0.02,
        # Adaptive temperature when entropy drifts below target
        enable_adaptive_temperature: bool = False,
        entropy_target_min: float = 2.0,
        temp_bump: float = 0.15,
        temp_max: float = 1.6,
        temp_bump_cooldown_steps: int = 3,
        early_diag_steps: int = 30,
    ):
        self.consecutive = consecutive
        self.entropy_action = entropy_action
        self.lr_reduction_factor = lr_reduction_factor
        self.dashboard = dashboard or MonitoringDashboard()
        self._lr_already_reduced = False
        self._entropy_zero_after_lr_reduce = 0

        self.enable_adaptive_beta = enable_adaptive_beta
        self.kl_near_zero_threshold = kl_near_zero_threshold
        self.stuck_beta = stuck_beta
        self._kl_near_zero_streak = 0
        self._beta_already_reduced = False
        self._original_beta: Optional[float] = None

        self.enable_adaptive_temperature = enable_adaptive_temperature
        self.entropy_target_min = entropy_target_min
        self.temp_bump = temp_bump
        self.temp_max = temp_max
        self.temp_bump_cooldown_steps = temp_bump_cooldown_steps
        self._last_temp_bump_step: int = -10_000
        self._temp_bumps: int = 0

        self.early_diag_steps = early_diag_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Force per-step logging during early diagnostics.
        if state.global_step <= self.early_diag_steps:
            control.should_log = True

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if not logs:
            return

        # Ignore intermediate logs that emit entropy=0 without a real loss/reward.
        is_optimizer_log = any(
            k in logs
            for k in (
                "loss",
                "rewards/format_reward_func/mean",
                "rewards/correctness_reward_func/mean",
                "train/loss",
            )
        )
        if not is_optimizer_log:
            return

        entropy = self._pick(logs, "entropy", "train/entropy")
        kl = self._pick(logs, "kl", "train/kl", "objective/kl")
        format_mean = self._pick(
            logs,
            "rewards/format_reward_func/mean",
            "train/rewards/format_reward_func/mean",
        )
        format_std = self._pick(
            logs,
            "rewards/format_reward_func/std",
            "train/rewards/format_reward_func/std",
        )
        clipped = self._pick(
            logs,
            "completions/clipped_ratio",
            "train/completions/clipped_ratio",
        )
        mean_len = self._pick(
            logs,
            "completions/mean_length",
            "train/completions/mean_length",
        )

        if state.global_step <= self.early_diag_steps:
            logger.info(
                "[early-diag] step=%s clipped_ratio=%s mean_length=%s entropy=%s kl=%s "
                "format_mean=%s",
                state.global_step,
                clipped,
                mean_len,
                entropy,
                kl,
                format_mean,
            )

        alerts = self.dashboard.record_live_signals(
            entropy=entropy,
            format_mean=format_mean,
            format_std=format_std,
            step=state.global_step,
            consecutive=self.consecutive,
        )

        if alerts["format_collapsed"]:
            logger.warning(
                "[stability] format_reward collapsed to 0 for %d consecutive logs "
                "(step=%s, clipped_ratio=%s, mean_len=%s). Likely systemic truncation "
                "before </answer> — reduce max_completion_length or check thinking loops.",
                self.consecutive,
                state.global_step,
                clipped,
                mean_len,
            )

        if alerts["entropy_collapsing"] and not alerts["entropy_zero"]:
            logger.warning(
                "[stability] entropy collapsing at step %s (entropy=%s). "
                "Consider raising temperature or reducing LR.",
                state.global_step,
                entropy,
            )

        if self.enable_adaptive_temperature:
            self._maybe_bump_temperature(
                entropy=entropy,
                entropy_collapsing=alerts["entropy_collapsing"] or alerts["entropy_zero"],
                step=state.global_step,
            )
        if self.enable_adaptive_beta:
            self._maybe_reduce_beta(
                kl=kl,
                entropy=entropy,
                entropy_collapsing=alerts["entropy_collapsing"],
                step=state.global_step,
            )

        if alerts["entropy_zero"]:
            self._handle_entropy_zero(args, state, control)

    def _maybe_bump_temperature(
        self,
        *,
        entropy: Optional[float],
        entropy_collapsing: bool,
        step: int,
    ) -> None:
        """Raise sampling temperature when entropy drifts below target (secular decay).

        Syncs TRL's trainer.temperature + generation_config so the next rollout
        actually uses the bump. Cooldown avoids oscillating every step.
        """
        if entropy is None:
            return
        soft = entropy < self.entropy_target_min
        hard = entropy_collapsing or entropy < 0.5
        if not (soft or hard):
            return
        if step - self._last_temp_bump_step < self.temp_bump_cooldown_steps:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "temperature"):
            if not getattr(self, "_warned_missing_trainer_temp", False):
                logger.error(
                    "[stability] adaptive temperature enabled but callback.trainer is unset "
                    "(or has no .temperature). Temp bumps will not apply."
                )
                self._warned_missing_trainer_temp = True
            return

        current = float(trainer.temperature)
        if current >= self.temp_max - 1e-9:
            return

        new_temp = min(self.temp_max, current + self.temp_bump)
        if new_temp <= current:
            return

        trainer.temperature = new_temp
        if hasattr(trainer, "args") and hasattr(trainer.args, "temperature"):
            trainer.args.temperature = new_temp
        gen_cfg = getattr(trainer, "generation_config", None)
        if gen_cfg is not None and hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = new_temp
        gen_kwargs = getattr(trainer, "generation_kwargs", None)
        if isinstance(gen_kwargs, dict):
            gen_kwargs["temperature"] = new_temp

        self._last_temp_bump_step = step
        self._temp_bumps += 1
        logger.warning(
            "[stability] entropy=%.4f below target %.2f (collapsing=%s) at step %s. "
            "Bumping temperature %.3f → %.3f (bump #%d, max=%.2f).",
            entropy,
            self.entropy_target_min,
            entropy_collapsing,
            step,
            current,
            new_temp,
            self._temp_bumps,
            self.temp_max,
        )

    def _maybe_reduce_beta(
        self,
        *,
        kl: Optional[float],
        entropy: Optional[float],
        entropy_collapsing: bool,
        step: int,
    ) -> None:
        """Temporarily lower KL beta when policy is stuck (KL≈0) but still exploring.

        Skipped entirely while entropy is collapsing or exactly zero — those need
        stronger reference pull / LR cuts, not a looser KL budget.
        """
        if self._beta_already_reduced:
            return
        if kl is None or entropy is None:
            return
        if entropy_collapsing or entropy == 0.0:
            self._kl_near_zero_streak = 0
            return

        if kl <= self.kl_near_zero_threshold:
            self._kl_near_zero_streak += 1
        else:
            self._kl_near_zero_streak = 0
            return

        if self._kl_near_zero_streak < self.consecutive:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "beta"):
            return

        current = float(trainer.beta)
        if current <= self.stuck_beta:
            return

        self._original_beta = current
        trainer.beta = self.stuck_beta
        # Keep GRPOConfig in sync when present.
        if hasattr(trainer, "args") and hasattr(trainer.args, "beta"):
            trainer.args.beta = self.stuck_beta
        self._beta_already_reduced = True
        logger.warning(
            "[stability] KL≈0 for %d consecutive logs at step %s while entropy=%.4f "
            "is healthy. Temporarily lowering beta %.4f → %.4f to unlock exploration.",
            self.consecutive,
            step,
            entropy,
            current,
            self.stuck_beta,
        )

    def _handle_entropy_zero(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        msg = (
            f"[stability] entropy exactly 0 for {self.consecutive}+ consecutive logs "
            f"at step {state.global_step} — policy exploration collapsed."
        )

        if self.entropy_action == "warn":
            logger.warning(msg)
            return

        if self.entropy_action == "stop":
            logger.error("%s Stopping training.", msg)
            control.should_training_stop = True
            return

        # reduce_lr: cut LR once; if zero entropy persists after that, hard-stop.
        if not self._lr_already_reduced:
            self._reduce_learning_rate(args)
            self._lr_already_reduced = True
            self._entropy_zero_after_lr_reduce = 0
            logger.error(
                "%s Reduced learning rate by factor %.2f. Will hard-stop if entropy "
                "stays at 0 for another %d consecutive zero-entropy alerts.",
                msg,
                self.lr_reduction_factor,
                self.consecutive,
            )
            return

        self._entropy_zero_after_lr_reduce += 1
        if self._entropy_zero_after_lr_reduce >= self.consecutive:
            logger.error(
                "%s Entropy still 0 after LR reduction. Stopping training.",
                msg,
            )
            control.should_training_stop = True

    def _reduce_learning_rate(self, args: TrainingArguments) -> None:
        args.learning_rate = args.learning_rate * self.lr_reduction_factor
        trainer = getattr(self, "trainer", None)
        optimizer = getattr(trainer, "optimizer", None) if trainer is not None else None
        if optimizer is not None:
            for group in optimizer.param_groups:
                if "lr" in group:
                    group["lr"] = group["lr"] * self.lr_reduction_factor
        schedulers = getattr(trainer, "lr_scheduler", None) if trainer is not None else None
        if schedulers is not None and hasattr(schedulers, "base_lrs"):
            schedulers.base_lrs = [
                lr * self.lr_reduction_factor for lr in schedulers.base_lrs
            ]

    @staticmethod
    def _pick(logs: dict, *keys: str) -> Optional[float]:
        for key in keys:
            if key in logs and logs[key] is not None:
                try:
                    return float(logs[key])
                except (TypeError, ValueError):
                    continue
        return None
