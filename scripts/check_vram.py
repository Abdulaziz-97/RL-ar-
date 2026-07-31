#!/usr/bin/env python3
"""Check GPU VRAM Usage and Active Processes.

Prints total, used, free VRAM (GiB) per GPU and lists any active GPU processes.
"""
import os
import subprocess
import torch

def format_gib(bytes_val: float) -> str:
    return f"{bytes_val / (1024 ** 3):.2f} GiB"

def main():
    print("=" * 80)
    print(" 🖥️  GPU VRAM & PROCESS MONITOR")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ PyTorch CUDA is not available on this machine.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} CUDA GPU(s):\n")

    for i in range(num_gpus):
        total = torch.cuda.get_device_properties(i).total_memory
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = total - reserved

        gpu_name = torch.cuda.get_device_name(i)
        util_pct = (reserved / total) * 100.0

        print(f"📍 GPU {i}: {gpu_name}")
        print(f"   • Total Memory    : {format_gib(total)}")
        print(f"   • Reserved/Used   : {format_gib(reserved)} ({util_pct:.1f}%)")
        print(f"   • Free VRAM       : {format_gib(free)}")
        print(f"   • PyTorch Alloc   : {format_gib(allocated)}")
        print("-" * 80)

    # Run nvidia-smi process list if available
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True
        ).strip()
        print("\n🔥 ACTIVE GPU PROCESSES:")
        if smi_out:
            for line in smi_out.splitlines():
                print(f"   • PID {line.strip()}")
        else:
            print("   • No active compute processes found (GPU VRAM 100% clean!).")
    except Exception:
        pass

    print("=" * 80)

if __name__ == "__main__":
    main()
