#!/usr/bin/env python3
"""Utility script to cleanly render Arabic text in terminal using python-bidi and arabic-reshaper."""
import sys
import json
from pathlib import Path

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_RENDERER = True
except ImportError:
    HAS_ARABIC_RENDERER = False

def print_arabic(text: str) -> str:
    if not text:
        return ""
    if HAS_ARABIC_RENDERER:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

def main():
    jsonl_path = Path("/workspace/RL-ar-/data/arabic_reasoning_rlvr_v4.jsonl")
    if not jsonl_path.exists():
        jsonl_path = Path("data/arabic_reasoning_rlvr_v4.jsonl")
    
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    
    if idx < 1 or idx > len(lines):
        print(f"Sample index {idx} out of range (1-{len(lines)})")
        return

    sample = lines[idx - 1]
    prompt = sample.get("prompt", "")
    
    print("=" * 80)
    print(f" 📖 ARABIC SAMPLE #{idx} (Domain: {sample.get('domain')}, Type: {sample.get('answer_spec', {}).get('type')})")
    print("=" * 80)
    print("PROMPT (Reshaped & Formatted for Terminal):")
    print(print_arabic(prompt))
    print("-" * 80)
    print("ANSWER SPEC:")
    print(json.dumps(sample.get("answer_spec"), ensure_ascii=False, indent=2))
    print("=" * 80)

if __name__ == "__main__":
    main()
