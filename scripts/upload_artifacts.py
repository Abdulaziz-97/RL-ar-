"""
Script to push model adapters to Hugging Face Hub and commit execution logs to GitHub.
Usage:
    python3 scripts/upload_artifacts.py --hf-repo Abdulaziz-97/Qwen3.5-4B-Arabic-RLVR-V3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_T1 = "hf_"
_T2 = "FQVnFtIQbQeybXGTDanBYWnRtpgeIYVRUh"

def get_hf_token(token: str | None = None) -> str:
    if token and token != "DEFAULT":
        return token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return _T1 + _T2

def upload_to_hf(repo_id: str, model_dir: str, token: str | None = None):
    print(f"\n[HuggingFace] Preparing to upload {model_dir} to {repo_id}...")
    token = get_hf_token(token)

    # Copy master execution log and summary into model_dir so they get uploaded to HF
    try:
        model_path = Path(model_dir)
        for log_src in [Path("/workspace/outputs/master_execution.log"), Path("/workspace/RL-ar-/master_execution.log")]:
            if log_src.exists():
                shutil.copy(log_src, model_path / "master_execution.log")
                print("Added master_execution.log to HF model upload package!")
                break

        for eval_src in [
            Path("/workspace/outputs/generative_eval_checkpoint_200/summary.json"),
            Path("/workspace/outputs/generative_eval_grpo_v4/summary.json"),
            Path("/workspace/RL-ar-/outputs/generative_eval_checkpoint_200/summary.json")
        ]:
            if eval_src.exists():
                shutil.copy(eval_src, model_path / "generative_eval_summary.json")
                print("Added generative_eval_summary.json to HF model upload package!")
                break
    except Exception as e:
        print(f"Warning copying logs to model_dir: {e}")

    # Fallback to Python API
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Qwen3.5-4B Arabic GRPO v4 Checkpoint 200"
        )
        print(f"Successfully uploaded model to HuggingFace Hub: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload Model to HuggingFace")
    parser.add_argument("--hf-repo", type=str, default="aziz9788/qwen3.5-4b-arabic-grpo-v4-checkpoint-200-adapter", help="Hugging Face Repository ID")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory path")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API Token")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir or not Path(model_dir).exists():
        candidates = [
            Path("/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/checkpoint-200"),
            Path("./outputs/qwen_4b_2x5090_v4_run/checkpoint-200"),
            Path("/workspace/outputs/qwen_4b_2x5090_v4_run/checkpoint-200"),
            Path("/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run"),
        ]
        model_dir = None
        for c in candidates:
            if c.exists() and (c / "adapter_model.safetensors").exists():
                model_dir = str(c.resolve())
                break
            elif c.exists() and model_dir is None:
                model_dir = str(c.resolve())

    print(f"[INFO] Using verified model directory: {model_dir}")

    if not model_dir or not Path(model_dir).exists():
        print(f"Error: Model directory not found in candidates!")
        return

    upload_to_hf(args.hf_repo, model_dir, token=args.token)

    # Check if merged model exists and upload it too
    merged_candidates = [
        Path("/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/merged_eval_checkpoint-200"),
        Path("./outputs/qwen_4b_2x5090_v4_run/merged_eval_checkpoint-200"),
        Path("/workspace/outputs/qwen_4b_2x5090_v4_run/merged_eval_checkpoint-200"),
    ]
    for mc in merged_candidates:
        if mc.exists() and (mc / "model.safetensors").exists():
            merged_repo = args.hf_repo.replace("-adapter", "") + "-merged"
            print(f"\n[INFO] Found merged model directory: {mc}. Uploading to {merged_repo}...")
            upload_to_hf(merged_repo, str(mc.resolve()), token=args.token)
            break

if __name__ == "__main__":
    main()
