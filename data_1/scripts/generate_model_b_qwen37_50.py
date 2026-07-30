import json
import urllib.request
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
prompts_file = PACK_ROOT / "outputs" / "benchmark_50_prompts.jsonl"
out_file = PACK_ROOT / "outputs" / "model_b_qwen37_50.jsonl"

openrouter_key = "YOUR_OPENROUTER_API_KEY_HERE"
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {openrouter_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
    "X-Title": "Arabic RLVR Pipeline"
}

def main():
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    done_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    done_ids.add(item.get("id"))

    print(f"[INFO] Generating Model B (Qwen 3.7 Flash via OpenRouter) responses ({len(done_ids)}/{len(prompts)} completed)...")

    with open(out_file, "a", encoding="utf-8") as out_f:
        for idx, p_item in enumerate(prompts, 1):
            if p_item["id"] in done_ids:
                continue
                
            prompt_text = p_item["prompt"]
            system_prompt = "أنت معلم رياضيات ومنطق خبير باللغة العربية الفصحى. اكتب خطوات التفكير التفصيلية خطوة بخطوة داخل وسم <think>...</think> وضعي الإجابة النهائية فقط داخل وسم <answer>...</answer>."
            user_content = f"{system_prompt}\n\nالمسألة:\n{prompt_text}"
            
            data = {
                "model": "qwen/qwen3.7-flash",
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.3
            }

            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    response_text = res["choices"][0]["message"]["content"]
                    
                    item = dict(p_item)
                    item["model"] = "qwen/qwen3.7-flash"
                    item["response"] = response_text
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"[Qwen 3.7 Flash {idx}/50] Generated successfully.")
            except Exception as e:
                print(f"[Qwen 3.7 Flash {idx}/50] Failed: {e}")
                
            time.sleep(0.4)

    print(f"[SUCCESS] Model B (Qwen 3.7 Flash) dataset complete at: {out_file}")

if __name__ == "__main__":
    main()
