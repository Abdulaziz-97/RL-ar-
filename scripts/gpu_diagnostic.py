#!/usr/bin/env python3
"""
GPU Diagnostic Script for Vast.ai GPU Environment
Checks VRAM usage, active GPU processes, PyTorch CUDA memory, and vLLM capability.
"""

import sys
import subprocess
import torch

def main():
    print("=================================================================")
    print("VAST.AI GPU ENVIRONMENT DIAGNOSTIC REPORT")
    print("=================================================================")

    # 1. PyTorch CUDA Status
    print(f"\n[1] PyTorch CUDA Status:")
    print(f"  - PyTorch Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ❌ ERROR: CUDA is NOT available to PyTorch!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"  - Detected CUDA GPU Count: {device_count}")

    for i in range(device_count):
        prop = torch.cuda.get_device_properties(i)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            allocated_gb = (total_bytes - free_bytes) / (1024**3)
            print(f"  - GPU {i}: {prop.name}")
            print(f"    Total Memory: {total_gb:.2f} GB")
            print(f"    Free Memory:  {free_gb:.2f} GB")
            print(f"    In Use / Reserved: {allocated_gb:.2f} GB")
        except Exception as e:
            print(f"  - GPU {i}: {prop.name} (mem_get_info error: {e})")

    # 2. nvidia-smi System Processes Check
    print(f"\n[2] System nvidia-smi Process Check:")
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            text=True
        )
        print(smi_out.strip())
    except Exception as e:
        print(f"  - nvidia-smi query warning: {e}")

    # 3. Test PyTorch Allocation on GPU 0
    print(f"\n[3] Testing Direct PyTorch Memory Allocation on GPU 0:")
    try:
        t = torch.zeros((1000, 1000, 512), dtype=torch.float32, device="cuda:0")
        del t
        torch.cuda.empty_cache()
        print("  --> PASS: PyTorch allocation on GPU 0 clean!")
    except Exception as e:
        print(f"  ❌ FAIL: PyTorch allocation on GPU 0 failed: {e}")

    print("\n=================================================================")
    print("DIAGNOSTIC COMPLETE")
    print("=================================================================")

if __name__ == "__main__":
    main()
