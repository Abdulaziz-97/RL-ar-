"""
Upload T06__qwen35-mixed-v6-lr1e5 LoRA adapter to Hugging Face Hub.
"""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding='utf-8')

LOCAL_T06_PATH = Path("I:/Instruction_following_team_handoff_extracted/Qwen3.5-4B/T06__qwen35-mixed-v6-lr1e5/lora/final")

def upload(repo_id: str, token: str):
    print("="*80)
    print(f" 🚀 UPLOADING T06 INSTRUCTION MODEL TO HUGGINGFACE: {repo_id}")
    print("="*80)
    
    api = HfApi(token=token)
    
    # 1. Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    print(f"[HF Hub] Created / Verified repository: https://huggingface.co/{repo_id}")
    
    # 2. Upload folder
    api.upload_folder(
        folder_path=str(LOCAL_T06_PATH),
        repo_id=repo_id,
        commit_message="Upload T06__qwen35-mixed-v6-lr1e5 SOTA Instruction Adapter",
    )
    print(f"\n✅ UPLOAD COMPLETE! Model is available at: https://huggingface.co/{repo_id}")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.getenv("HF_TOKEN") or input("Enter your HuggingFace Token (hf_...): ")
    
    repo_id = os.getenv("HF_REPO_ID", "Abdulaziz-97/T06__qwen35-mixed-v6-lr1e5")
    upload(repo_id, token.strip())
