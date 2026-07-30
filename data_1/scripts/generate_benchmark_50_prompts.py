import json
import random
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from vendor.synth.programmatic import GENERATORS

def main():
    rng = random.Random(999) # Fixed seed for clean comparison
    domains = ["gsm8k", "math", "math_comp", "logic", "arapro_knowledge", "ifeval_multiconstraint", "aratrust_truth"]
    domain_weights = [0.30, 0.25, 0.20, 0.15, 0.04, 0.03, 0.03]

    out_file = PACK_ROOT / "outputs" / "benchmark_50_prompts.jsonl"
    out_file.parent.mkdir(exist_ok=True)

    prompts = []
    seen = set()

    while len(prompts) < 50:
        dom = rng.choices(domains, weights=domain_weights, k=1)[0]
        gen = GENERATORS[dom]
        s = gen(rng)
        
        p = s.prompt.strip()
        if p not in seen:
            seen.add(p)
            prompts.append({
                "id": len(prompts) + 1,
                "domain": dom,
                "prompt": p,
                "gold": str(s.ground_truth),
                "solution_steps": s.solution_steps
            })

    with open(out_file, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Generated 50 benchmark evaluation prompts at: {out_file}")

if __name__ == "__main__":
    main()
