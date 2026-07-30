import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

for path in ROOT.rglob("*.py"):
    if ".venv" in str(path) or ".git" in str(path):
        continue
    try:
        content = path.read_text(encoding="utf-8")
        cleaned = re.sub(r'sk-or-v1-[a-f0-9]{64}', 'YOUR_OPENROUTER_API_KEY_HERE', content)
        if cleaned != content:
            path.write_text(cleaned, encoding="utf-8")
            print(f"Sanitized secret in: {path}")
    except Exception:
        pass
