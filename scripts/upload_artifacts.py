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

def upload_to_hf(repo_id: str, model_dir: str):
    print(f"\n[HuggingFace] Preparing to upload {model_dir} to {repo_id}...")
    token = os.environ.get("HF_TOKEN")

    # Copy master execution log and summary into model_dir so they get uploaded to HF
    try:
        model_path = Path(model_dir)
        for log_src in [Path("/workspace/outputs/master_execution.log"), Path("/workspace/RL-ar-/master_execution.log")]:
            if log_src.exists():
                shutil.copy(log_src, model_path / "master_execution.log")
                print("Added master_execution.log to HF model upload package!")
                break

        for eval_src in [Path("/workspace/outputs/generative_eval_grpo_v3/summary.json"), Path("/workspace/RL-ar-/outputs/generative_eval_grpo_v3/summary.json")]:
            if eval_src.exists():
                shutil.copy(eval_src, model_path / "generative_eval_summary.json")
                print("Added generative_eval_summary.json to HF model upload package!")
                break
    except Exception as e:
        print(f"Warning copying logs to model_dir: {e}")

    # Try using new 'hf' CLI tool first
    if shutil.which("hf"):
        try:
            if token:
                subprocess.run(["hf", "auth", "login", "--token", token], check=True)
            subprocess.run(["hf", "upload", repo_id, model_dir, "."], check=True)
            print(f"Successfully uploaded model adapters and logs using 'hf' CLI to: https://huggingface.co/{repo_id}")
            return
        except Exception as e:
            print(f"hf CLI upload failed ({e}), falling back to Python API...")

    # Fallback to Python API
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload GRPO_V3 SOTA Arabic Reasoning Model Adapters"
        )
        print(f"Successfully uploaded model adapters to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")

def commit_logs_to_github():
    print("\n[GitHub] Committing training and evaluation logs...")
    logs_dir = Path("outputs_logs")
    logs_dir.mkdir(exist_ok=True)

    # Set git author identity if not configured
    try:
        subprocess.run(["git", "config", "user.email", "abdulaziz@saudi-llm.ai"], check=False)
        subprocess.run(["git", "config", "user.name", "Abdulaziz"], check=False)
    except Exception:
        pass

    # Copy key logs if they exist
    src_master = Path("/workspace/outputs/master_execution.log")
    if not src_master.exists():
        src_master = Path("/workspace/RL-ar-/master_execution.log")

    if src_master.exists():
        shutil.copy(src_master, logs_dir / "master_execution.log")
        print("Copied master_execution.log to outputs_logs/")

    src_eval = Path("/workspace/outputs/generative_eval_grpo_v3/summary.json")
    if not src_eval.exists():
        src_eval = Path("/workspace/RL-ar-/outputs/generative_eval_grpo_v3/summary.json")

    if src_eval.exists():
        shutil.copy(src_eval, logs_dir / "generative_eval_summary.json")
        print("Copied generative_eval_summary.json to outputs_logs/")

    try:
        subprocess.run(["git", "add", "outputs_logs/"], check=True)
        subprocess.run(["git", "commit", "-m", "docs(logs): add GRPO_V3 training and master evaluation execution logs"], check=True)
        subprocess.run(["git", "push", "origin", "Efficient-Arabic-Reasnoning-Pipeline"], check=True)
        print("Successfully committed and pushed logs to GitHub repository!")
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload Model to HuggingFace and Logs to GitHub")
    parser.add_argument("--hf-repo", type=str, default="aziz9788/qwen3.5-4b-arabic-grpo-v3", help="Hugging Face Repository ID")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory path")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir or not Path(model_dir).exists():
        candidates = [
            Path("/workspace/RL-ar-/outputs/qwen_4b_2x5090_v3_run/checkpoint-105"),
            Path("/workspace/RL-ar-/outputs/qwen_4b_2x5090_v3_run"),
            Path("./outputs/qwen_4b_2x5090_v3_run/checkpoint-105"),
            Path("./outputs/qwen_4b_2x5090_v3_run"),
            Path("/workspace/outputs/qwen_4b_2x5090_v3_run/checkpoint-105"),
            Path("/workspace/outputs/qwen_4b_2x5090_v3_run"),
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

    upload_to_hf(args.hf_repo, model_dir)
    commit_logs_to_github()

if __name__ == "__main__":
    main()
