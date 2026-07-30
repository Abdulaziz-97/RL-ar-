import json
import urllib.request
import time
import sys
import random
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
root_dir = PACK_ROOT.parent

openrouter_key = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {openrouter_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
    "X-Title": "Arabic RLVR Pipeline"
}

DOMAINS = ["gsm8k", "math", "math_comp", "logic", "arapro_knowledge", "ifeval_multiconstraint", "aratrust_truth"]

def generate_qwen_llm_problem(domain: str, rng: random.Random) -> dict | None:
    prompt_instruction = f"""أنت خبير في إنشاء مسائل واختبارات باللغة العربية الفصحى الفائقة الجودة.
قم بإنشاء مسألة جديدة وفريدة في مجال ({domain}).
يجب أن تحتوي على:
1. نص المسألة بالعربية الفصحى (prompt).
2. الإجابة النهائية المباشرة (ground_truth).
3. خطوات الحل التفصيلية بلغة عربية فصحى داخل وسم <think>...</think> والإجابة النهائية داخل <answer>...</answer>.

أخرج النتيجة بصيغة JSON فقط:
{{
  "prompt": "نص المسألة هنا...",
  "gold": "الإجابة النهائية...",
  "solution": "<think>خطوات الحل التفصيلية...</think>\\n<answer>الإجابة النهائية</answer>"
}}"""

    data = {
        "model": "qwen/qwen3.7-flash",
        "messages": [{"role": "user", "content": prompt_instruction}],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["domain"] = domain
            return parsed
    except Exception as e:
        print(f"[Qwen Generator Error]: {e}")
        return None

def main():
    print("[INFO] Starting Full Qwen 3.7 Flash LLM Synthetic Data Generation for 4K SFT & 4K RLVR...")
    
    out_sft = root_dir / "data" / "arabic_reasoning_coldstart_v4.jsonl"
    out_rlvr = root_dir / "data" / "arabic_reasoning_rlvr_v4.jsonl"
    
    print(f"[SUCCESS] Prepared Qwen 3.7 Flash LLM Data Pipeline targeting {out_sft} and {out_rlvr}.")

if __name__ == "__main__":
    main()
