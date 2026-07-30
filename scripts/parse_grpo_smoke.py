"""Parse GRPO smoke log for monitor checks."""
from __future__ import annotations

import re
import sys
from pathlib import Path

log = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/grpo_v6_smoke/run.log")
text = log.read_text(encoding="utf-8", errors="replace")

# Full trainer log lines that include reward breakdowns
pat = re.compile(r"\{'loss':[^\n]*rewards/correctness_reward_func/mean[^\n]*\}")
keys = [
    ("corr", "rewards/correctness_reward_func/mean"),
    ("fmt", "rewards/format_reward_func/mean"),
    ("lang", "rewards/language_reward_func/mean"),
    ("reward", "reward"),
    ("kl", "kl"),
    ("ent", "entropy"),
    ("len", "completions/mean_length"),
    ("clip", "completions/clipped_ratio"),
    ("epoch", "epoch"),
]

print("=== step metrics ===")
rows = []
for i, m in enumerate(pat.finditer(text), 1):
    s = m.group(0)
    row = {"i": i}
    for short, k in keys:
        mm = re.search(rf"'{re.escape(k)}': '([^']+)'", s)
        row[short] = mm.group(1) if mm else "-"
    rows.append(row)
    print(
        f"{i:2d} corr={row['corr']:>7} fmt={row['fmt']:>7} lang={row['lang']:>7} "
        f"kl={row['kl']:>7} ent={row['ent']:>7} reward={row['reward']:>9} epoch={row['epoch']}"
    )

ps = re.findall(r"\|\s*(\d+)/(?:20|30|50)\s*\[", text)
print("progress", ps[-5:] if ps else None)
print("log_bytes", log.stat().st_size)
if rows:
    last = rows[-1]
    try:
        kl_v = float(last["kl"]) if last["kl"] not in ("-", "", None) else 0.0
        ent_v = float(last["ent"]) if last["ent"] not in ("-", "", None) else 99.0
        hint = "WATCH" if kl_v > 3 or ent_v < 1.5 else "OK"
    except ValueError:
        hint = "OK"
    print("verdict_hint", hint)
