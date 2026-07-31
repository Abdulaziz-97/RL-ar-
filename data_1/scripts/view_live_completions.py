#!/usr/bin/env python3
"""Live Arabic Completion Viewer for Terminal.

Parses latest log completions and renders them in full wide format
using python-bidi and arabic-reshaper without narrow table wrapping.
"""
import os
import sys
import json
import glob
from pathlib import Path

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except ImportError:
    HAS_ARABIC = False

def render_text(text: str) -> str:
    if not text:
        return ""
    if HAS_ARABIC:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

def main():
    print("=" * 80)
    print(" 📖 LIVE ARABIC GRPO COMPLETIONS VIEWER (WIDE TERMINAL FORMAT)")
    print("=" * 80)

    # Search for latest wandb / trl log output or master log
    log_file = Path("/workspace/outputs/master_execution_v4.log")
    if not log_file.exists():
        log_file = Path("outputs/master_execution_v4.log")

    if not log_file.exists():
        print("Log file not found.")
        return

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    completion_lines = [l for l in lines if "<think>" in l or "<answer>" in l or "rewards/" in l]

    if not completion_lines:
        print("No completion lines logged yet. Check back in a few steps!")
        return

    print("--- LATEST GENERATED ARABIC COMPLETION (WIDE FORMAT) ---")
    for line in completion_lines[-30:]:
        clean = line.strip()
        print(render_text(clean))
    print("=" * 80)

if __name__ == "__main__":
    main()
