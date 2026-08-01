"""
Push each Trainer checkpoint to Hugging Face Hub on_save.

Trainer evaluates before save when eval_steps == save_steps. This callback
stashes on_evaluate metrics, writes them into the local checkpoint as
eval_results.json, then uploads under hub path
{hub_checkpoint_name_prefix}-{step} (default: checkpoint-evaluated-{step}).

Enables early deletion of local checkpoints after a successful push so Vast
disk stays under limit. Fail soft: never delete if the hub push fails.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve HF token from explicit arg or env (never hardcode secrets)."""
    if explicit:
        return explicit
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _checkpoint_dir(output_dir: str, step: int) -> Path:
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    if ckpt.is_dir():
        return ckpt
    # Final / best-model style saves may land directly in output_dir.
    return Path(output_dir)


def _list_local_checkpoints(output_dir: str) -> list[tuple[int, Path]]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except ValueError:
            continue
        found.append((step, child))
    found.sort(key=lambda x: x[0])
    return found


def _json_safe(obj: Any) -> Any:
    """Make Trainer metrics JSON-serializable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        # numpy / torch scalars
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return str(obj)


class HubCheckpointCallback(TrainerCallback):
    """Upload evaluated checkpoint-{step}/ to hub as checkpoint-evaluated-{step}."""

    def __init__(
        self,
        hub_model_id: str,
        *,
        enabled: bool = True,
        hub_private: bool = True,
        delete_local_after_push: bool = True,
        keep_local_last_n: Optional[int] = None,
        token: Optional[str] = None,
        hub_checkpoint_name_prefix: str = "checkpoint-evaluated",
    ):
        self.hub_model_id = hub_model_id
        self.enabled = enabled
        self.hub_private = hub_private
        self.delete_local_after_push = delete_local_after_push
        # None / <=0 → keep only the latest pushed checkpoint locally after push.
        self.keep_local_last_n = keep_local_last_n
        self._token = token
        self.hub_checkpoint_name_prefix = (
            hub_checkpoint_name_prefix or "checkpoint-evaluated"
        ).rstrip("/")
        self._pushed_steps: set[int] = set()
        self._repo_ready = False
        # Trainer runs evaluate() before save when strategies align; stash metrics.
        self._last_eval_step: Optional[int] = None
        self._last_eval_metrics: Optional[dict[str, Any]] = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.enabled:
            return
        if getattr(args, "process_index", 0) != 0:
            return
        step = int(state.global_step)
        self._last_eval_step = step
        self._last_eval_metrics = dict(metrics or {})
        print(
            f"[HUB-CKPT] Step {step}: captured eval metrics "
            f"({len(self._last_eval_metrics)} keys) for upcoming checkpoint push.",
            flush=True,
        )
        return

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        if getattr(args, "process_index", 0) != 0:
            return
        if not self.hub_model_id:
            print(
                "[HUB-CKPT] push_checkpoints_to_hub=true but hub_model_id is empty; skipping.",
                flush=True,
            )
            return

        step = int(state.global_step)
        output_dir = getattr(args, "output_dir", "./outputs")
        ckpt_dir = _checkpoint_dir(output_dir, step)
        if not ckpt_dir.is_dir():
            print(
                f"[HUB-CKPT] Step {step}: checkpoint dir not found ({ckpt_dir}); skipping push.",
                flush=True,
            )
            return

        self._write_eval_results(ckpt_dir, step)
        ok = self._push_checkpoint(ckpt_dir, step)
        if ok:
            self._pushed_steps.add(step)
            if self.delete_local_after_push:
                self._delete_older_local_checkpoints(output_dir)
        return

    def _hub_path_in_repo(self, step: int) -> str:
        return f"{self.hub_checkpoint_name_prefix}-{step}"

    def _write_eval_results(self, ckpt_dir: Path, step: int) -> None:
        """Persist eval metrics next to adapter weights before Hub upload."""
        payload: dict[str, Any] = {
            "global_step": step,
            "hub_path_in_repo": self._hub_path_in_repo(step),
            "evaluated": False,
            "metrics": {},
        }
        if self._last_eval_metrics is not None and self._last_eval_step == step:
            payload["evaluated"] = True
            payload["metrics"] = _json_safe(self._last_eval_metrics)
        elif self._last_eval_metrics is not None:
            # Save without matching eval (e.g. end-of-run save); still attach last eval.
            payload["evaluated"] = False
            payload["last_eval_step"] = self._last_eval_step
            payload["metrics"] = _json_safe(self._last_eval_metrics)
            print(
                f"[HUB-CKPT] Step {step}: no same-step eval (last_eval_step="
                f"{self._last_eval_step}); attaching last metrics for reference.",
                flush=True,
            )
        else:
            print(
                f"[HUB-CKPT] Step {step}: no eval metrics yet; pushing weights only.",
                flush=True,
            )

        out = ckpt_dir / "eval_results.json"
        try:
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[HUB-CKPT] Wrote {out.name} (evaluated={payload['evaluated']})", flush=True)
        except Exception as e:
            print(f"[HUB-CKPT] Warning: failed to write eval_results.json: {e}", flush=True)

    def _ensure_repo(self, api, token: str) -> None:
        if self._repo_ready:
            return
        api.create_repo(
            repo_id=self.hub_model_id,
            exist_ok=True,
            private=self.hub_private,
            token=token,
            repo_type="model",
        )
        self._repo_ready = True

    def _push_checkpoint(self, ckpt_dir: Path, step: int) -> bool:
        token = resolve_hf_token(self._token)
        if not token:
            print(
                "[HUB-CKPT] ERROR: HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set; "
                "cannot push checkpoint. Local checkpoint retained.",
                flush=True,
            )
            return False

        path_in_repo = self._hub_path_in_repo(step)
        print(
            f"\n[HUB-CKPT] Step {step}: pushing {ckpt_dir} → "
            f"{self.hub_model_id}/{path_in_repo} (private={self.hub_private})...",
            flush=True,
        )
        try:
            from huggingface_hub import HfApi, login

            login(token=token, add_to_git_credential=False)
            api = HfApi(token=token)
            self._ensure_repo(api, token)
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=self.hub_model_id,
                repo_type="model",
                path_in_repo=path_in_repo,
                commit_message=(
                    f"Upload evaluated checkpoint {path_in_repo} "
                    f"(step={step}, with eval_results.json)"
                ),
                token=token,
            )
            print(
                f"[HUB-CKPT] Successfully pushed {path_in_repo} → "
                f"https://huggingface.co/{self.hub_model_id}/tree/main/{path_in_repo}",
                flush=True,
            )
            return True
        except Exception as e:
            print(
                f"[HUB-CKPT] ERROR: push failed for {path_in_repo}: {e}. "
                "Local checkpoint retained (no delete).",
                flush=True,
            )
            logger.exception("Hub checkpoint push failed at step %s", step)
            return False

    def _delete_older_local_checkpoints(self, output_dir: str) -> None:
        """Delete pushed local checkpoints older than the latest keep_local_last_n."""
        keep_n = self.keep_local_last_n
        if keep_n is None or keep_n <= 0:
            keep_n = 1

        checkpoints = _list_local_checkpoints(output_dir)
        if len(checkpoints) <= keep_n:
            return

        # Keep the newest keep_n; delete older ones only if successfully pushed.
        to_keep = {step for step, _ in checkpoints[-keep_n:]}
        for step, path in checkpoints:
            if step in to_keep:
                continue
            if step not in self._pushed_steps:
                print(
                    f"[HUB-CKPT] Keeping unpushed local {path.name} (push never confirmed).",
                    flush=True,
                )
                continue
            try:
                shutil.rmtree(path)
                print(
                    f"[HUB-CKPT] Deleted local {path.name} after successful hub push "
                    f"(keeping latest {keep_n}).",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[HUB-CKPT] Warning: failed to delete local {path}: {e}",
                    flush=True,
                )
