import json
import urllib.request
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
prompts_file = PACK_ROOT / "outputs" / "benchmark_50_prompts.jsonl"
out_file = PACK_ROOT / "outputs" / "model_a_deepseek_50.jsonl"

api_key = "YOUR_API_KEY_HERE"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def main():
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    print(f"[INFO] Generating Model A (DeepSeek Pro) responses for {len(prompts)} prompts...")
    results = []

    for idx, p_item in enumerate(prompts, 1):
        prompt_text = p_item["prompt"]
        system_prompt = "أنت معلم رياضيات ومنطق خبير باللغة العربية الفصحى. اكتب خطوات التفكير التفصيلية خطوة بخطوة داخل وسم <think>...</think> وضعي الإجابة النهائية فقط داخل وسم <answer>...</answer>."
        
        user_content = f"{system_prompt}\n\nالمسألة:\n{prompt_text}"
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.3
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                response_text = res["choices"][0]["message"]["content"]
                
                item = dict(p_item)
                item["model"] = "DeepSeek-Pro"
                item["response"] = response_text
                results.append(item)
                print(f"[DeepSeek {idx}/50] Generated successfully.")
        except Exception as e:
            print(f"[DeepSeek {idx}/50] Failed: {e}")
            
        time.sleep(0.5)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(results)} Model A (DeepSeek Pro) samples to: {out_file}")

if __name__ == "__main__":
    main()
