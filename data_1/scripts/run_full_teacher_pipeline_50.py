import json
import os
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

import dspy
from dotenv import load_dotenv

load_dotenv(PACK_ROOT / ".env")

from vendor.synth.dspy_teacher import ArabicTeacher

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
    
    print("[INFO] Configuring DSPy with DeepSeek Pro teacher backend...")
    lm = dspy.LM(
        model="openai/deepseek-chat",
        api_key=api_key,
        api_base="https://api.deepseek.com/v1",
        temperature=0.3
    )
    dspy.configure(lm=lm)

    teacher = ArabicTeacher()

    prompts_file = PACK_ROOT / "outputs" / "benchmark_50_prompts.jsonl"
    out_file = PACK_ROOT / "outputs" / "full_teacher_deepseek_pro_50.jsonl"

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    print(f"[INFO] Running FULL 8-Stage ArabicTeacher Pipeline over {len(prompts)} prompts...")
    results = []

    for idx, p in enumerate(prompts, 1):
        domain = p.get("domain", "gsm8k")
        prompt_text = p.get("prompt", "")
        gold = str(p.get("gold", ""))

        print(f"[{idx}/50] Executing 8-stage teacher pipeline (plan -> solve -> refine -> materialize -> polish)...")
        try:
            res = teacher(domain=domain, problem=prompt_text, ground_truth=gold)
            response_text = res.response if hasattr(res, "response") else str(res)
            
            item = dict(p)
            item["teacher_model"] = "deepseek-v4-pro-full-pipeline"
            item["response"] = response_text
            results.append(item)
            print(f"[{idx}/50] SUCCESS: Teacher completed all 8 pipeline stages.")
        except Exception as e:
            print(f"[{idx}/50] FAILED: {e}")

    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(results)} full teacher pipeline samples to: {out_file}")

if __name__ == "__main__":
    main()
